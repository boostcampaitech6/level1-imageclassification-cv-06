import torch
from rembg import remove
from rembg import new_session
from PIL import Image
from tqdm import tqdm
from enum import Enum
import os
​
​
class MaskLabels(int, Enum):
    """마스크 라벨을 나타내는 Enum 클래스"""
​
    MASK = 0
    INCORRECT = 1
    NORMAL = 2
​
​
_file_names = {
    "incorrect_mask": MaskLabels.INCORRECT,
    "normal": MaskLabels.NORMAL,
}
​
data_dir = '/data/ephemeral/home/data/train/images'
​
profiles = os.listdir(data_dir)
​
for profile in tqdm(profiles, desc="Processing", unit="Folder"):
    if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
        continue
​
    img_folder = os.path.join(data_dir, profile)
    for file_name in os.listdir(img_folder):
        _file_name, ext = os.path.splitext(file_name)
        if (
            _file_name not in _file_names
        ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
            continue
​
        img_path = os.path.join(
            data_dir, profile, file_name
        )
​
        output_path = os.path.join(
            data_dir, profile, (_file_name + '_horizontal_fliped' + ext)
        )
​
        input = Image.open(img_path)
        output = input.transpose(Image.FLIP_LEFT_RIGHT)
        output.save(output_path)
​
# import onnxruntime as ort
​
# providers = ort.get_available_providers()
# print("Available Providers:", providers)