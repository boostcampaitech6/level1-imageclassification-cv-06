import torch
import torch.nn as nn
import sys
import os

# ensemble.py가 있는 폴더의 경로를 추가
folder_path = '/home/level1/code/cv6/models'
sys.path.append(folder_path)

from mask_model import resnet50, ViT16  # 모델 정의가 있는 파일

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def forward(self, x):
        """
        일반 평균 방식
        """
        # # 각 모델의 예측 결과를 저장
        # outputs = [model(x) for model in self.models]

        # # 예측 결과들의 평균을 계산
        # avg_output = torch.mean(torch.stack(outputs), dim=0)
        # return avg_output
        """voting 방식"""
        # 각 모델의 예측 결과를 저장
        preds = [model(x) for model in self.models]

        preds = torch.stack(preds, dim=1)

        votes = preds.argmax(dim=-1)
        final_pred, _ = votes.mode(dim=1)

        return final_pred

def load_model(model_cls, num_classes, model_path, device):
    model = model_cls(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)  # 모델을 해당 장치로 이동
    model.eval()
    return model

def create_ensemble(models_info, num_classes, device):
    models = []
    for model_name, model_path in models_info.items():
        if model_name == 'resnet50':
            models.append(load_model(resnet50, num_classes, model_path, device))
        elif model_name == 'ViT16':
            models.append(load_model(ViT16, num_classes, model_path, device))
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    return EnsembleModel(models)
