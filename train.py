import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion
import torch.nn.functional as F

from sklearn.metrics import f1_score


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def grid_image(np_images, gts, preds, n=16, shuffle=False, iscutmix=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(
        figsize=(12, 18 + 2)
    )  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(
        top=0.8
    )  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n**0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        if args.one_hot:
            if iscutmix:
                _, gt = gts[choice].topk(2, 0, True, True)
                _, pred = preds[choice].topk(2, 0, True, True)
                gt = gt.tolist()
                pred = pred.tolist()
            else:
                gt = gts[choice]
                pred = preds[choice]
            image = np_images[choice]
            title = "\n".join([f"gt: {gt}, pred: {pred}"])
        else:
            gt = gts[choice].item()
            pred = preds[choice].item()
            image = np_images[choice]
            gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
            pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
            title = "\n".join(
                [
                    f"{task} - gt: {gt_label}, pred: {pred_label}"
                    for gt_label, pred_label, task in zip(
                        gt_decoded_labels, pred_decoded_labels, tasks
                    )
                ]
            )

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def run(args, train_set, val_set):
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    return train_loader, val_loader


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(
        os.path.join(model_dir, args.model_type + "_" + args.name)
    )

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(
        import_module("dataset"), args.dataset
    )  # default: MaskBaseDataset
    dataset = dataset_module(data_dir=data_dir, one_hot=args.one_hot)
    iscutmix = "cutmix" in args.dataset.lower()
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(
        import_module("dataset"), args.augmentation
    )  # default: BaseAugmentation
    transform = transform_module(args, dataset)
    dataset.set_transform(transform)

    # -- data_loader
    k_fold_type = args.k_fold_type
    k_fold = args.k_fold
    if k_fold_type == 0:
        splits = [dataset.split_dataset()]
    elif k_fold_type == 1:
        splits = dataset.split_dataset2(k_fold)
    elif k_fold_type == 2:
        splits = dataset.split_dataset3(k_fold)

    # -- model
    model_module = getattr(
        import_module("models." + args.model_type), args.model
    )  # default: BaseModel
    model = model_module(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(
        args.criterion, classes=dataset.num_classes
    )  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD

    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.1)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    best_val_f1 = 0
    iter = 1

    for train_set, val_set in splits:
        ### 학습 매 fold마다 독립적으로 진행되도록
        model = model_module(num_classes=num_classes).to(device)
        model = torch.nn.DataParallel(model)
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4,
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        # print("h")
        # print(train_set, val_set)
        train_loader, val_loader = run(args, train_set, val_set)
        print(str(iter) + "번째 fold")
        iter += 1

        for epoch in range(args.epochs):
            # train loop
            model.train()
            # print("hh")
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()

                if args.one_hot:
                    if iscutmix:
                        _, pred = outs.topk(2, 1, True, True)
                        _, labels = labels.topk(2, 1, True, True)
                        correct = pred.eq(labels)

                        matches += correct.all(dim=1).sum().item()
                    else:
                        matches += (
                            (torch.argmax(outs, dim=-1) == torch.argmax(labels, dim=-1))
                            .sum()
                            .item()
                        )
                else:
                    if outs.dim() != 1:
                        outs = torch.argmax(outs, dim=-1)
                    matches += (outs == labels).sum().item()

                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch + 1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4f} || training accuracy {train_acc:4.2%} || lr {current_lr}",
                        end="\r",
                    )
                    logger.add_scalar(
                        "Train/loss", train_loss, epoch * len(train_loader) + idx
                    )
                    logger.add_scalar(
                        "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                    )

                    loss_value = 0
                    matches = 0

            scheduler.step()
            print()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                val_f1_items = []  # F1점수 저장 리스트

                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    if args.eval_f1:
                        preds = torch.argmax(outs, dim=-1)
                        # F1 점수 계산
                        f1_item = f1_score(
                            labels.cpu().numpy(), preds.cpu().numpy(), average="macro"
                        )
                        val_f1_items.append(f1_item)

                    loss_item = criterion(outs, labels).item()

                    if args.one_hot:
                        if iscutmix:
                            _, pred = outs.topk(2, 1, True, True)
                            _, topklabels = labels.topk(2, 1, True, True)
                            correct = pred.eq(topklabels)

                            acc_item = correct.all(dim=1).sum().item()

                        else:
                            acc_item = (
                                (
                                    torch.argmax(outs, dim=-1)
                                    == torch.argmax(labels, dim=-1)
                                )
                                .sum()
                                .item()
                            )

                    else:
                        if outs.dim() != 1:
                            outs = torch.argmax(outs, dim=-1)
                        acc_item = (labels == outs).sum().item()

                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = (
                            torch.clone(inputs)
                            .detach()
                            .cpu()
                            .permute(0, 2, 3, 1)
                            .numpy()
                        )
                        inputs_np = dataset_module.denormalize_image(
                            inputs_np, dataset.mean, dataset.std
                        )
                        if args.one_hot:
                            figure = grid_image(
                                inputs_np,
                                labels,
                                outs,
                                n=16,
                                shuffle=args.dataset != "MaskSplitByProfileDataset",
                                iscutmix=iscutmix,
                            )
                        else:
                            figure = grid_image(
                                inputs_np,
                                labels,
                                outs,
                                n=16,
                                shuffle=args.dataset != "MaskSplitByProfileDataset",
                            )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                best_val_loss = min(best_val_loss, val_loss)

                if args.eval_f1:
                    val_f1 = np.mean(val_f1_items)  # 평균 F1 점수 계산

                if val_acc > best_val_acc:
                    print(
                        f"New best model for val accuracy : {val_acc:4.2%}. saving the best model.."
                    )

                    if args.eval_f1:
                        print(f"val f1 : {val_f1:.4f}!")

                    torch.save(
                        model.module.state_dict(),
                        f"{save_dir}/{args.model_type}_best.pth",
                    )
                    best_val_acc = val_acc

                    if args.eval_f1:
                        best_val_f1 = val_f1

                torch.save(
                    model.module.state_dict(), f"{save_dir}/{args.model_type}_last.pth"
                )

                print(f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || ", end="")

                if args.eval_f1:
                    print(f"[Val] F1 Score: {val_f1:4.2} || ", end="")

                print(
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2} \n"
                )

                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)

                if args.eval_f1:
                    logger.add_scalar("Val/f1", val_f1, epoch)

                logger.add_figure("results", figure, epoch)
                print()


def str2bool(v):
    """
        argument로 True, False 값을 받아오기위한 함수

    Args:
        v (str): true, false와 같은 문자열

    Returns:
        bool : True or False
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MaskBaseDataset",
        help="dataset augmentation type (default: MaskBaseDataset)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="BaseAugmentation",
        help="data augmentation type (default: BaseAugmentation)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=[128, 96],
        help="resize size for image when training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=1000,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--model", type=str, default="BaseModel", help="model type (default: BaseModel)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="SGD", help="optimizer type (default: SGD)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="ratio for validaton (default: 0.2)",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="cross_entropy",
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=20,
        help="learning rate scheduler deacy step (default: 20)",
    )
    parser.add_argument(
        "--lr_decay_gamma",
        type=float,
        default=0.5,
        help="gamma value scheduler deacy step (default: 0.5)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--name", default="exp", help="model save at {SM_MODEL_DIR}/{name}"
    )

    # Container environment
    parser.add_argument(
        "--data_dir",
        type=str,
        # default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train/images"),
        # 이름이 SM_CHANNEL_TRAIN인 이유는, 아마존 SAGE MAKER에서 모델을 학습할 때 해당 환경변수 이름으로 Path를 지정하면 알아서 모델을 학습시켜 준다고 한다.
        default=os.environ.get("SM_CHANNEL_TRAIN", "~/data/train/images"),
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model")
    )

    parser.add_argument(
        "--model_type",
        type=str,
        help="you have to choose which task you will train",
        default="age_model",
    )

    parser.add_argument(
        "--k_fold",
        type=int,
        help="k for (Stratified) K-fold Cross Validation",
        default=5,
    )

    parser.add_argument(
        "--k_fold_type",
        type=int,
        help="0: No K-fold, 1: K-fold, 2: Stratified K-fold",
        default=0,
    )

    parser.add_argument(
        "--one_hot",
        type=str2bool,
        help="for dataset labels, True for one-hot type, False for scalar",
        default=False,
    )

    parser.add_argument(
        "--eval_f1",
        type=str2bool,
        help="use f1 to evaluate model",
        default=False,
    )

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
