import torch
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from tqdm import tqdm
from importlib import import_module
import multiprocessing

# 데이터 위치
data_dir = "/data/ephemeral/home/data/train/images"

# dataset 임포트
from dataset import AgeModelDataset, MaskModelDataset, GenderModelDataset

# Augmentation 임포트
from dataset import CustomAugmentation

# dataset 불러오기


dataset = AgeModelDataset(data_dir)


class MyArg:
    resize = [128, 96]


# transform 임포트
transform = CustomAugmentation(MyArg(), dataset)
dataset.set_transform(transform)

# load image from dataloader
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    # num_workers=multiprocessing.cpu_count() // 2,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

# model import
from models.age_model import ResNet50

# model 불러오기

## Fill in ##
num_classes = 3
model = ResNet50(num_classes)
#############

## Fill in ##
model_loc = [
    "/data/ephemeral/home/project/repo/level1-imageclassification-cv-06/gender_model_gender-exp-12-20-data1-ResNet50-300epoch-lr0.0001-batch256/gender_model_best.pth"
]  # 여기에 model.pth경로 (여러개 가능!)
#############
# 1"/data/ephemeral/home/project/repo/level1-imageclassification-cv-06/age_model_exp-12-20-data1-ResNet50-300epoch-lr0.0001-batch256/age_model_best.pth"


print("batch 개수: ", len(data_loader))
print("Dataset 개수: ", len(dataset))

for loc in model_loc:
    model.load_state_dict(torch.load(loc, map_location="cuda"))
    model.eval()
    labels = []
    preds = []
    """
        ※주의사항 ※
        load image from dataloader를 계속 불러버리면, 계속 데이터 셋이 늘어나고, 시간이 오래 걸린다!!!!
    """
    for index, batch in enumerate(tqdm(data_loader)):
        image, label = batch
        image = image
        label = label.long()
        labels.extend(label.numpy())
        pred = model(image)
        preds.extend(torch.argmax(pred, dim=-1))
        # 아래 주석을 사용하면 빠른 결과, 부정확한 score. (약 1분 소요)
        # 그렇지 않으면 느린 결과, 정확한 score. (약 10분 소요)
        # if index > 20:
        #     break

    recall = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average="macro")
    conf_matrix = confusion_matrix(labels, preds)

    print("Recall:", recall, "  ")
    print("f1:", f1, "  ")
    print("Confusion Matrix:")
    print(conf_matrix)

    print("copy&paste to MarkDown")
    if num_classes == 2:
        print("||0|1|  ")
        print("|---:|:---:|:---:|  ")
        for i in range(num_classes):
            print(f"|{i}|{conf_matrix[i][0]}|{conf_matrix[i][1]}|  ")
    else:
        print("||0|1|2|  ")
        print("|---:|:---:|:---:|:---:|  ")
        for i in range(num_classes):
            print(
                f"|{i}|{conf_matrix[i][0]}|{conf_matrix[i][1]}|{conf_matrix[i][2]}|  "
            )
