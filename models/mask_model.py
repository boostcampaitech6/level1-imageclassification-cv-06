import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BaseModel(nn.Module):
    """
    기본적인 컨볼루션 신경망 모델
    """

    def __init__(self, num_classes):
        """
        모델의 레이어 초기화

        Args:
            num_classes (int): 출력 레이어의 뉴런 수
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            x (torch.Tensor): num_classes 크기의 출력 텐서
        """
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet34 = models.resnet34(pretrained=True)
        in_features = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(in_features, num_classes)

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.resnet34(x)
        return x


class Resnet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet34 = models.resnet34(pretrained=True)
        in_features = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(in_features, num_classes)


    def forward(self, x):
        x = self.resnet34(x)
        return x
    

class MaskCustomModel(nn.Module):
    def __init__(self, num_classes, download = False):
        super().__init__()

        self.model_wear = Resnet34(2)
        self.model_correct = Resnet34(2)

        if download:
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            self.wear_path = './best_model/mask_model_wear_notwear.pth'
            self.correct_path = './best_model/mask_model_correct_incorrect.pth'
            # 모델 가중치를 로드한다.
            self.model_wear.load_state_dict(torch.load(self.wear_path, map_location=device))
            self.model_correct.load_state_dict(torch.load(self.correct_path, map_location=device))


    def forward(self, x):
        x1 = self.model_wear(x)
        x1 = torch.argmax(x1, -1)

        for i in range(len(x1)):
            if x1[i] == 1: #쓴 경우
                x2 = self.model_correct(x[i].unsqueeze(0))
                x2 = torch.argmax(x2, -1)
                if x2[0] == 0: #마스크가 정상인 경우
                    x1[i] = 0
                else:
                    x1[i] = 1
            else: #안쓴 경우
                x1[i] = 2
        return x1