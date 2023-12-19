import torch
from rembg import remove
from rembg import new_session
from PIL import Image
from tqdm import tqdm
from enum import Enum
import os
from PIL import ImageEnhance


class AgeLabels(int, Enum):
    """나이 라벨을 나타내는 Enum 클래스"""

    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        """숫자로부터 해당하는 나이 라벨을 찾아 반환하는 클래스 메서드"""
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskLabels(int, Enum):
    """마스크 라벨을 나타내는 Enum 클래스"""

    MASK = 0
    INCORRECT = 1
    NORMAL = 2


_file_names = {
    "mask1": MaskLabels.MASK,
    "mask2": MaskLabels.MASK,
    "mask3": MaskLabels.MASK,
    "mask4": MaskLabels.MASK,
    "mask5": MaskLabels.MASK,
    "incorrect_mask": MaskLabels.INCORRECT,
    "normal": MaskLabels.NORMAL,
}

data_dir = "/data/ephemeral/home/data/train/images"

profiles = os.listdir(data_dir)

for profile in tqdm(profiles, desc="Processing", unit="Folder"):
    if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
        continue

    # last_two_digits = int(profile[-2:])

    # if last_two_digits >= 60:
    img_folder = os.path.join(data_dir, profile)
    for file_name in os.listdir(img_folder):
        _file_name, ext = os.path.splitext(file_name)
        if _file_name not in _file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
            continue

        img_path = os.path.join(
            data_dir, profile, file_name
        )  # (resized_data, 000004_male_Asian_54, mask1.jpg)

        id, gender, race, age = profile.split("_")
        age_label = AgeLabels.from_number(age)

        if age_label == AgeLabels.OLD:
            # output_path = os.path.join(
            #     data_dir, profile, (_file_name + '_horizontal_fliped' + ext)
            # )

            input = Image.open(img_path)
            output = input.transpose(Image.FLIP_LEFT_RIGHT)
            output.save(
                os.path.join(
                    data_dir, profile, (_file_name + "_horizontal_fliped" + ext)
                )
            )

            enhancer = ImageEnhance.Sharpness(input)
            sharpened_img = enhancer.enhance(
                2.0
            )  # enhance factor 2.0 means sharpened image
            sharpened_img.save(
                os.path.join(data_dir, profile, (_file_name + "_sharpened" + ext))
            )


# import onnxruntime as ort

# providers = ort.get_available_providers()
# print("Available Providers:", providers)
