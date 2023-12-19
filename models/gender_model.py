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

class ResNet50Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Model, self).__init__()
        
        # ResNet50 기반의 사전 학습된 모델 불러오기 224 224가 인풋
        self.base_model = torchvision.models.resnet50(pretrained=True)
        
        # ResNet50의 마지막 fc layer를 변경하여 원하는 num_classes로 출력 차원 설정
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
    
class ResNet34Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34Model, self).__init__()
        
        # ResNet34 기반의 사전 학습된 모델 불러오기 224 224가 인풋
        self.base_model = torchvision.models.resnet34(pretrained=True)
        
        # ResNet34의 마지막 fc layer를 변경하여 원하는 num_classes로 출력 차원 설정
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
    
class VGG16Custom(nn.Module):
    def __init__(self, num_classes):
        super(VGG16Custom, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        num_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.vgg(x)
        return x

class InceptionV3Custom(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3Custom, self).__init__()
        self.inception = torchvision.models.inception_v3(pretrained=True)
        num_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        outputs = self.inception(x)
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs
    
class MobileNetV2Custom(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2Custom, self).__init__()
        self.mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.mobilenet(x)
        return x

class EfficientNetb0Custom(nn.Module):
    # input size 224 224
    def __init__(self, num_classes):
        super(EfficientNetb0Custom,self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        in_features = self.efficientnet.classifier.in_features
        new_fc_layer = nn.Linear(in_features, num_classes)

        # 모델의 마지막 FC 레이어를 새로운 FC 레이어로 교체
        self.efficientnet.classifier = new_fc_layer
    
    def forward(self, x):
        x = self.efficientnet(x)
        return x

