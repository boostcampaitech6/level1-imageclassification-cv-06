import os
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm
from enum import Enum
import numpy as np

class MaskLabels(int, Enum):
    """마스크 라벨을 나타내는 Enum 클래스"""
    MASK = 0
    INCORRECT = 1
    NORMAL = 2

_file_names = {
    "incorrect_mask": MaskLabels.INCORRECT,
    "normal": MaskLabels.NORMAL,
}

data_dir = '/home/level1/train/images'
profiles = os.listdir(data_dir)

def augment_image_aug1(image):
    """aug1 증강: 가로 반전, 색상 변경, 랜덤 회전"""
    # 가로 반전
    image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 색상 변경
    color_variations = random.uniform(0.5, 1.5)
    image = ImageEnhance.Color(image).enhance(color_variations)

    # 랜덤 회전
    angle = random.choice([30, 45, 60, 75])
    image = image.rotate(angle)

    return image

def add_noise(image):
    """이미지에 랜덤 노이즈를 추가하는 함수"""
    # PIL 이미지를 NumPy 배열로 변환
    np_image = np.array(image)

    # 노이즈 생성
    noise = np.random.normal(0, 25, np_image.shape)

    # 이미지에 노이즈 추가
    np_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)

    # NumPy 배열을 다시 PIL 이미지로 변환
    return Image.fromarray(np_image)

def augment_image_aug2(image):
    """aug2 증강: 수직 반전, 밝기 조정, 랜덤 회전"""

    # 밝기 조정
    brightness_variations = random.uniform(0.5, 1.5)
    image = ImageEnhance.Brightness(image).enhance(brightness_variations)

    # 노이즈 추가
    image = add_noise(image)

    # 랜덤 회전
    angle = random.choice([30, 45, 60, 75])
    image = image.rotate(angle)

    return image

for profile in tqdm(profiles, desc="Processing", unit="Folder"):
    if profile.startswith("."):
        continue

    img_folder = os.path.join(data_dir, profile)
    for file_name in os.listdir(img_folder):
        _file_name, ext = os.path.splitext(file_name)
        if _file_name not in _file_names:
            continue

        img_path = os.path.join(data_dir, profile, file_name)
        input_image = Image.open(img_path)

        # aug1 적용 및 저장
        augmented_image_aug1 = augment_image_aug1(input_image)
        augmented_image_path_aug1 = os.path.join(
            data_dir, profile, (_file_name + '_aug1' + ext)
        )
        augmented_image_aug1.save(augmented_image_path_aug1)

        # aug2 적용 및 저장
        augmented_image_aug2 = augment_image_aug2(input_image)
        augmented_image_path_aug2 = os.path.join(
            data_dir, profile, (_file_name + '_aug2' + ext)
        )
        augmented_image_aug2.save(augmented_image_path_aug2)
