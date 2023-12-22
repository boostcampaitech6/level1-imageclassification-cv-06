import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from torchvision.models import densenet121


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
        x = self.fc(x)
        return torch.softmax(x, dim=-1)


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
        return torch.softmax(x, dim=-1)


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)
        # self.fc = nn.Linear(in_features // 2, num_classes)

    def forward(self, x):
        x = self.resnet50(x)
        # x = self.fc(x)
        return torch.softmax(x, dim=-1)


# class ResNet50(nn.Module):
#     def __init__(self, num_classes):
#         super(ResNet50, self).__init__()
#         self.resnet50 = models.resnet50(pretrained=True)

#         # Freeze the layers
#         for param in self.resnet50.parameters():
#             param.requires_grad = False

#         # Replace the last fully connected layer
#         in_features = self.resnet50.fc.in_features
#         self.resnet50.fc = nn.Linear(in_features, num_classes)

#     def forward(self, x):
#         x = self.resnet50(x)
#         return torch.softmax(x, dim=-1)


class ViT(nn.Module):
    def __init__(self, num_classes):
        super(ViT, self).__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=True, img_size=(128, 96)
        )

        # Replace the classifier layer
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        x = self.vit(x)
        return torch.softmax(x, dim=-1)


class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.effnet = timm.create_model("tf_efficientnet_b0_ns", pretrained=True)

        # Replace the classifier layer
        self.effnet.classifier = nn.Linear(
            self.effnet.classifier.in_features, num_classes
        )

    def forward(self, x):
        x = self.effnet(x)
        return torch.softmax(x, dim=-1)


class ModifiedEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedEfficientNet, self).__init__()
        self.effnet = timm.create_model("tf_efficientnet_b0_ns", pretrained=True)

        # Replace the classifier layer with a new one with Dropout and an additional FC layer
        self.effnet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.effnet.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.effnet(x)
        return torch.softmax(x, dim=-1)


class ModifiedResNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the last fc layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.resnet(x)
        return torch.softmax(x, dim=-1)


class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet, self).__init__()
        self.densenet = densenet121(pretrained=True)

        # 분류기 레이어를 교체합니다.
        self.densenet.classifier = nn.Linear(
            self.densenet.classifier.in_features, num_classes
        )

    def forward(self, x):
        x = self.densenet(x)
        return torch.softmax(x, dim=-1)
