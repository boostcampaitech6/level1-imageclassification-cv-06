import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import AgeModelDataset, GenderModelDataset, MaskModelDataset, TestDataset


def load_model(saved_model, model_type, num_classes, device):
    """
    저장된 모델의 가중치를 로드하는 함수입니다.

    Args:
        saved_model (str): 모델 가중치가 저장된 디렉토리 경로
        num_classes (int): 모델의 클래수 수
        device (torch.device): 모델이 로드될 장치 (CPU 또는 CUDA)

    Returns:
        model (nn.Module): 가중치가 로드된 모델
    """

    model_cls = getattr(import_module(f"models.{model_type}"), getattr(args, model_type))
    model = model_cls(num_classes=num_classes)

    # 모델 가중치를 로드한다.
    model.load_state_dict(torch.load(saved_model, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    모델 추론을 수행하는 함수

    Args:
        data_dir (str): 테스트 데이터가 있는 디렉토리 경로
        model_dir (str): 모델 가중치가 저장된 디렉토리 경로
        output_dir (str): 결과 CSV를 저장할 디렉토리 경로
        args (argparse.Namespace): 커맨드 라인 인자

    Returns:
        None
    """

    # CUDA를 사용할 수 있는지 확인
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 클래스의 개수를 설정한다. (마스크, 성별, 나이의 조합으로 총 Combination은 18)
    age_num_classes = AgeModelDataset.num_classes  # 3
    gender_num_classes = GenderModelDataset.num_classes  # 2
    mask_num_classes = MaskModelDataset.num_classes  # 3
    
    age_model = load_model(f'{model_dir}/age_model_best.pth', 'age_model', age_num_classes, device).to(device)
    gender_model = load_model(f'{model_dir}/gender_model_best.pth', 'gender_model', gender_num_classes, device).to(device)
    mask_model = load_model(f'{model_dir}/mask_model_best.pth', 'mask_model', mask_num_classes, device).to(device)

    age_model.eval()
    gender_model.eval()
    mask_model.eval()


    # 이미지 파일 경로와 정보 파일을 읽어온다.
    img_root = os.path.join(data_dir, "images")
    info_path = os.path.join(data_dir, "info.csv")
    info = pd.read_csv(info_path)

    # 이미지 경로를 리스트로 생성한다.
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]


    age_dataset = TestDataset(img_paths, args)
    transform_module = getattr(
        import_module("dataset"), args.age_augmentation
    )  # default: BaseAugmentation
    transform = transform_module(args, age_dataset)
    age_dataset.set_transform(transform)


    gender_dataset = TestDataset(img_paths, args)
    transform_module = getattr(
        import_module("dataset"), args.gender_augmentation
    )  # default: BaseAugmentation
    transform = transform_module(args, gender_dataset)
    gender_dataset.set_transform(transform)


    mask_dataset = TestDataset(img_paths, args)
    transform_module = getattr(
        import_module("dataset"), args.mask_augmentation
    )  # default: BaseAugmentation
    transform = transform_module(args, mask_dataset)
    mask_dataset.set_transform(transform)


    age_loader = torch.utils.data.DataLoader(
        age_dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    gender_loader = torch.utils.data.DataLoader(
        gender_dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    mask_loader = torch.utils.data.DataLoader(
        mask_dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Start calculating inference results..")
    age_preds = []
    gender_preds = []
    mask_preds = []

    with torch.no_grad():
        print("Inferencing age...")
        for idx, images in enumerate(age_loader):
            images = images.to(device)
            pred = age_model(images)
            pred = pred.argmax(dim=-1)
            age_preds.extend(pred.cpu().numpy())
        
        print("Inferencing for age complete. Inferencing gender...")
        for idx, images in enumerate(gender_loader):
            images = images.to(device)
            pred = gender_model(images)
            pred = pred.argmax(dim=-1)
            gender_preds.extend(pred.cpu().numpy())

        print("Inferencing for gender complete. Inferencing mask...")
        for idx, images in enumerate(mask_loader):
            images = images.to(device)
            pred = mask_model(images)
            mask_preds.extend(pred.cpu().numpy())

        print("Inferencing Complete!")
    
    preds = []
    for i in range(len(age_dataset)):
        # 3개의 출력결과를 합쳐 18개의 Class로 통합해주는 부분
        preds.append(mask_preds[i]*6+gender_preds[i]*3+age_preds[i])
    
    # 예측 결과를 데이터프레임에 저장하고 csv 파일로 출력한다.
    info["ans"] = preds
    save_path = os.path.join(output_dir, f"output.csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == "__main__":
    # 커맨드 라인 인자를 파싱한다.
    parser = argparse.ArgumentParser()

    # 데이터와 모델 체크포인트 디렉터리 관련 인자
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=(96, 128),
        help="resize size for image when you trained (default: (96, 128))",
    )
    parser.add_argument(
        "--age_model", type=str, default="BaseModel", help="age model type (default: BaseModel)"
    )

    parser.add_argument(
        "--gender_model", type=str, default="BaseModel", help="gender model type (default: BaseModel)"
    )

    parser.add_argument(
        "--mask_model", type=str, default="BaseModel", help="mask model type (default: BaseModel)"
    )

    parser.add_argument(
        "--age_augmentation", type=str, default="BaseAugmentation", help="age augmentation type (default: BaseAugmentation)"
    )

    parser.add_argument(
        "--gender_augmentation", type=str, default="BaseAugmentation", help="gender augmentation type (default: BaseAugmentation)"
    )

    parser.add_argument(
        "--mask_augmentation", type=str, default="BaseAugmentation", help="mask augmentation type (default: BaseAugmentation)"
    )

    # 컨테이너 환경 변수
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_EVAL", "/data/ephemeral/home/data/eval/images"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_MODEL", "./best_model"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./"),
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # 모델 추론을 수행한다.
    inference(data_dir, model_dir, output_dir, args)
