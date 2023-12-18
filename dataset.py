import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import (
    Resize,
    ToTensor,
    Normalize,
    Compose,
    CenterCrop,
    ColorJitter,
)
from copy import copy
import torch.nn.functional as F

# 지원되는 이미지 확장자 리스트
IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    """파일 이름이 이미지 확장자를 가지는지 확인하는 함수

    Args:
        filename (str): 확인하고자 하는 파일 이름

    Returns:
        bool: 파일 이름이 이미지 확장자를 가지면 True, 그렇지 않으면 False.
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    """
    기본적인 Augmentation을 담당하는 클래스

    Attributes:
        transform (Compose): 이미지를 변환을 위한 torchvision.transforms.Compose 객체
    """

    def __init__(self, args, dataset):
        """
        Args:
            args (arguments): CLI로 받아온 argument.
            dataset (Dataset) :
        """
        self.transform = Compose(
            [
                Resize(args.resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=dataset.mean, std=dataset.std),
            ]
        )

    def __call__(self, image):
        """
        이미지에 저장된 transform 적용

        Args:
            Image (PIL.Image): Augumentation을 적용할 이미지

        Returns:
            Tensor: Argumentation이 적용된 이미지
        """
        return self.transform(image)


class CustomSharpenTransform:
    """
    이미지를 날카롭게 만드는 Transform
    """

    def __init__(self, factor=2.5):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(self.factor)
        return img


class SharpenAugmentation:
    """
    Image를 날카롭게 만드는 Augmentation.
    기본적인 Normalize도 포함되어 있다.
    """

    def __init__(self, args, dataset):
        self.transform = Compose(
            [
                Resize(args.resize, Image.BILINEAR),
                CustomSharpenTransform(),
                ToTensor(),
                Normalize(mean=dataset.mean, std=dataset.std),
            ]
        )

    def __call__(self, image):
        """
        이미지에 저장된 transform 적용

        Args:
            Image (PIL.Image): Augumentation을 적용할 이미지

        Returns:
            Tensor: Argumentation이 적용된 이미지
        """
        return self.transform(image)


class AddGaussianNoise(object):
    """이미지에 Gaussian Noise를 추가하는 클래스"""

    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class CustomAugmentation:
    """커스텀 Augmentation을 담당하는 클래스"""

    def __init__(self, agrs, dataset):
        self.transform = Compose(
            [
                CenterCrop((320, 256)),
                Resize(agrs.resize, Image.BILINEAR),
                ColorJitter(0.1, 0.1, 0.1, 0.1),
                ToTensor(),
                Normalize(mean=dataset.mean, std=dataset.std),
                AddGaussianNoise(),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class CutMixVerticalAugmentation:
    def __init__(self, percentage=0.5):
        """
        origin_image가 얼마나 많은 비율을 갖고 있을지를 정하는 percentage.
        percentage가 0.5면 세로로 절반, 0.6이면 오리지널이 60%, 다른 랜덤한 사진이 40%를 갖는다.
        """
        self.percentage = percentage

    def __call__(self, origin_image, origin_label, cutmix_image, cutmix_label):
        return self.cutmix(origin_image, origin_label, cutmix_image, cutmix_label)

    def cutmix(self, origin_image, origin_label, cutmix_image, cutmix_label):
        """_summary_
            cutmix를 적용하는 함수.
            origin_image가 self.percentage 만큼, random하게 선택된 cutmix_image가 1-self.percentage만큼
            합성되는 이미지를 반환한다.
        Args:
            origin_image (PIL.JpegImagePlugin.JpegImageFile): PIL.Image로 불러온 이미지
            origin_label (torch.Tensor): origin_image에 맞는 one-hot vector label
            cutmix_image (PIL.JpegImagePlugin.JpegImageFile): PIL.Image로 불러온 이미지
            cutmix_label (torch.Tensor): cutmix_image에 맞는 one-hot vector label

        Returns:
            mixed_image (PIL.Image.Image): mix된 이미지
            mixed_label (torch.Tensor): mix된 이미지의 mix된 label
        """
        mixed_image = copy(origin_image)
        width, height = mixed_image.size

        bbx1, bby1, bbx2, bby2 = self.vertical_bbox(width, height)

        cutmix_image = cutmix_image.crop((bbx1, bby1, bbx2, bby2))

        mixed_image.paste(cutmix_image, (bbx1, bby1))
        mixed_label = origin_label * self.percentage + cutmix_label * (
            1 - self.percentage
        )
        return mixed_image, mixed_label

    def vertical_bbox(self, width, height):
        # Generate random bounding box coordinates for CutMix
        cut_w = int(width * self.percentage)
        cut_h = int(height)

        # Randomly choose the top-left corner of the bounding box
        cx = 0
        cy = 0

        # Calculate the bounding box coordinates
        # 그저 수직으로 자르는것이기 때문에, 다음과 같이 구성했다.
        bbx1 = max(0, cx + cut_w)
        bby1 = max(0, cy - cut_h)
        bbx2 = width
        bby2 = height

        return bbx1, bby1, bbx2, bby2


class MaskLabels(int, Enum):
    """마스크 라벨을 나타내는 Enum 클래스"""

    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    """성별 라벨을 나타내는 Enum 클래스"""

    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        """문자열로부터 해당하는 성별 라벨을 찾아 반환하는 클래스 메서드"""
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(
                f"Gender value should be either 'male' or 'female', {value}"
            )


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


class MaskBaseDataset(Dataset):
    """마스크 데이터셋의 기본 클래스

    image:  PIL.Image
    labels: Long
    ex) (PIL.Image 타입의 어떤 이미지), (해당 이미지에 맞는 class num (0~18 중 1))
    """

    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
    ):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()  # 데이터셋을 설정
        self.calc_statistics()  # 통계시 계산 (평균 및 표준 편차)

    def setup(self):
        """데이터 디렉토리로부터 이미지 경로와 라벨을 설정하는 메서드"""
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if (
                    _file_name not in self._file_names
                ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        """데이터셋의 통계치를 계산하는 메서드"""
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine"
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    def set_transform(self, transform):
        """변환(transform)을 설정하는 메서드"""
        self.transform = transform

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        """데이터셋의 길이를 반환하는 메서드"""
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        """인덱스에 해당하는 마스크 라벨을 반환하는 메서드"""
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        """인덱스에 해당하는 성별 라벨을 반환하는 메서드"""
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        """인덱스에 해당하는 나이 라벨을 반환하는 메서드"""
        return self.age_labels[index]

    def read_image(self, index):
        """인덱스에 해당하는 이미지를 읽는 메서드"""
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        """다중 라벨을 하나의 클래스로 인코딩하는 메서드"""
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(
        multi_class_label,
    ) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        """인코딩된 다중 라벨을 각각의 라벨로 디코딩하는 메서드"""
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        """정규화된 이미지를 원래대로 되돌리는 메서드"""
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """데이터셋을 학습과 검증용으로 나누는 메서드
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여 torch.utils.data.Subset 클래스 둘로 나눕니다.
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
    train / val 나누는 기준을 이미지에 대해서 random 이 아닌 사람(profile)을 기준으로 나눕니다.
    구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다.
    이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
    ):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        """프로필을 학습과 검증용으로 나누는 메서드"""
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.sample(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {"train": train_indices, "val": val_indices}

    def setup(self):
        """데이터셋 설정을 하는 메서드. 프로필 기준으로 나눈다."""
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if (
                        _file_name not in self._file_names
                    ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(
                        self.data_dir, profile, file_name
                    )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        """프로필 기준으로 나눈 데이터셋을 Subset 리스트로 반환하는 메서드"""
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class TestDataset(Dataset):
    """테스트 데이터셋 클래스"""

    def __init__(
        self, img_paths, args, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
    ):
        self.img_paths = img_paths
        self.mean = mean
        self.std = std
        # self.transform = Compose(
        #     [
        #         Resize(args.resize, Image.BILINEAR),
        #         ToTensor(),
        #         Normalize(mean=mean, std=std),
        #     ]
        # )

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        """데이터셋의 길이를 반환하는 메서드"""
        return len(self.img_paths)

    def set_transform(self, transform):
        self.transform = transform


class MaskModelDataset(Dataset):
    """
    마스크 데이터셋의 기본 클래스

    image:  PIL.Image
    labels: Long
    ex) (PIL.Image 타입의 어떤 이미지), (해당 이미지에 맞는 class num (0~3 중 1))

    """

    num_classes = 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    image_paths = []
    mask_labels = []

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
        one_hot=False,
    ):
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.mean = mean
        self.std = std
        self.transform = None
        self.one_hot = one_hot
        self.setup()  # 데이터셋을 설정
        self.calc_statistics()

    def setup(self):
        """데이터 디렉토리로부터 이미지 경로와 라벨을 설정하는 메서드"""
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if (
                    _file_name not in self._file_names
                ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)

    def set_transform(self, transform):
        """변환(transform)을 설정하는 메서드"""
        self.transform = transform

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        image_transform = self.transform(image)
        return image_transform, mask_label

    def __len__(self):
        """데이터셋의 길이를 반환하는 메서드"""
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        """인덱스에 해당하는 마스크 라벨을 반환하는 메서드"""
        if self.one_hot:
            return F.one_hot(
                torch.tensor(self.mask_labels[index] * 1), self.num_classes
            ).float()
        return self.mask_labels[index]

    def read_image(self, index):
        """인덱스에 해당하는 이미지를 읽는 메서드"""
        image_path = self.image_paths[index]
        return Image.open(image_path)

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """데이터셋을 학습과 검증용으로 나누는 메서드
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여 torch.utils.data.Subset 클래스 둘로 나눕니다.
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set

    def calc_statistics(self):
        """데이터셋의 통계치를 계산하는 메서드"""
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine"
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    @staticmethod
    def denormalize_image(image, mean, std):
        """정규화된 이미지를 원래대로 되돌리는 메서드"""
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp


class CutMixMaskModelDataset(Dataset):
    """MaskModel에 사용하기 위한 CutMix를 적용하는 데이터셋 클래스

    Args:
        data_dir (str): 데이터 위치
        mean
        std
        val_ratio
        cutmix_augmentation (CutMixVerticalAugmentation): CutMix 수행하는 Class
        cutmix_prob (float): CutMix를 수행 할 확률

    """

    num_classes = 3

    image_paths = []
    mask_labels = []

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
        cutmix_prob=0.49,
        one_hot=True,
    ):
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.mean = mean
        self.std = std
        self.transform = None
        self.cutmix_augmentation = CutMixVerticalAugmentation(cutmix_prob)
        self.setup()  # 데이터셋을 설정
        self.calc_statistics()

    def setup(self):
        """데이터 디렉토리로부터 이미지 경로와 라벨을 설정하는 메서드"""
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if (
                    _file_name not in self._file_names
                ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)

    def set_transform(self, transform):
        """변환(transform)을 설정하는 메서드"""
        self.transform = transform

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        # assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        origin_image = self.read_image(index)
        origin_mask_label = self.get_mask_label(index)

        cutmix_index = random.randint(0, len(self.image_paths) - 1)
        cutmix_image = self.read_image(cutmix_index)
        cutmix_age_label = self.get_mask_label(cutmix_index)
        augmented_img, augmented_label = self.cutmix_augmentation(
            origin_image, origin_mask_label, cutmix_image, cutmix_age_label
        )
        if self.transform:
            image_transform = self.transform(augmented_img)
            return image_transform, augmented_label

        # 라벨로 one-hot vector 리턴.
        return augmented_img, augmented_label

    def __len__(self):
        """데이터셋의 길이를 반환하는 메서드"""
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        """인덱스에 해당하는 마스크 라벨을 반환하는 메서드"""
        return F.one_hot(torch.tensor(self.mask_labels[index] * 1), self.num_classes)

    def read_image(self, index):
        """인덱스에 해당하는 이미지를 읽는 메서드"""
        image_path = self.image_paths[index]
        return Image.open(image_path)

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """데이터셋을 학습과 검증용으로 나누는 메서드
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여 torch.utils.data.Subset 클래스 둘로 나눕니다.
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set

    def calc_statistics(self):
        """데이터셋의 통계치를 계산하는 메서드"""
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine"
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    @staticmethod
    def denormalize_image(image, mean, std):
        """정규화된 이미지를 원래대로 되돌리는 메서드"""
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp


class AgeModelDataset(Dataset):
    """
    나이 데이터셋의 기본 클래스

    image:  PIL.Image
    labels: Long
    ex) (PIL.Image 타입의 어떤 이미지), (해당 이미지에 맞는 class num (0~3 중 1))

    """

    num_classes = 3

    image_paths = []
    age_labels = []

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
        one_hot=False,
    ):
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.mean = mean
        self.std = std
        self.transform = None
        self.one_hot = one_hot
        self.setup()  # 데이터셋을 설정
        self.calc_statistics()

    def setup(self):
        """데이터 디렉토리로부터 이미지 경로와 라벨을 설정하는 메서드"""
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if (
                    _file_name not in self._file_names
                ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  # (resized_data, 000004_male_Asian_54, mask1.jpg)

                id, gender, race, age = profile.split("_")
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.age_labels.append(age_label)

    def set_transform(self, transform):
        """변환(transform)을 설정하는 메서드"""
        self.transform = transform

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        age_label = self.get_age_label(index)
        image_transform = self.transform(image)
        return image_transform, age_label

    def __len__(self):
        """데이터셋의 길이를 반환하는 메서드"""
        return len(self.image_paths)

    def get_age_label(self, index) -> AgeLabels:
        """인덱스에 해당하는 나이 라벨을 반환하는 메서드"""
        if self.one_hot:
            return F.one_hot(
                torch.tensor(self.age_labels[index] * 1), self.num_classes
            ).float()
        return self.age_labels[index]

    def read_image(self, index):
        """인덱스에 해당하는 이미지를 읽는 메서드"""
        image_path = self.image_paths[index]
        return Image.open(image_path)

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """데이터셋을 학습과 검증용으로 나누는 메서드
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여 torch.utils.data.Subset 클래스 둘로 나눕니다.
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set

    def calc_statistics(self):
        """데이터셋의 통계치를 계산하는 메서드"""
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine"
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    @staticmethod
    def denormalize_image(image, mean, std):
        """정규화된 이미지를 원래대로 되돌리는 메서드"""
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp


class CutMixAgeModelDataset(Dataset):
    """AgeModel에 사용하기 위한 CutMix를 적용하는 데이터셋 클래스

    Args:
        data_dir (str): 데이터 위치
        mean
        std
        val_ratio
        cutmix_augmentation (CutMixVerticalAugmentation): CutMix 수행하는 Class
        cutmix_prob (float): CutMix를 수행 할 확률
        use_skewed (bool): 데이터가 skewed되어 있다면 skewe된 데이터만 mix하게 결정하는 flag
        skew_prob (float): use_skewd가 True인 경우, 특정 확률로 skew된 데이터를 mix하도록 함.
    """

    num_classes = 3

    image_paths = []
    age_labels = []
    skewed_image_paths = []
    skewed_age_labels = []

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
        cutmix_prob=0.49,
        one_hot=True,
        use_skewed=True,
        skew_prob=0.6,
    ):
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.mean = mean
        self.std = std
        self.transform = None
        self.cutmix_augmentation = CutMixVerticalAugmentation(cutmix_prob)
        self.use_skewed = use_skewed
        self.skew_prob = skew_prob
        self.setup()  # 데이터셋을 설정
        self.calc_statistics()

    def setup(self):
        """데이터 디렉토리로부터 이미지 경로와 라벨을 설정하는 메서드"""
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if (
                    _file_name not in self._file_names
                ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                id, gender, race, age = profile.split("_")
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.age_labels.append(age_label)

                if age_label == AgeLabels.OLD:
                    self.skewed_image_paths.append(img_path)
                    self.skewed_age_labels.append(age_label)

    def set_transform(self, transform):
        """변환(transform)을 설정하는 메서드"""
        self.transform = transform

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        # assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        origin_image = self.read_image(index)
        origin_age_label = self.get_age_label(index)

        if self.use_skewed and random.random() < self.skew_prob:
            cutmix_index = random.randint(0, len(self.skewed_image_paths) - 1)
            cutmix_image = self.read_image(cutmix_index, self.use_skewed)
            cutmix_age_label = self.get_age_label(cutmix_index, self.use_skewed)

        else:
            cutmix_index = random.randint(0, len(self.image_paths) - 1)
            cutmix_image = self.read_image(cutmix_index)
            cutmix_age_label = self.get_age_label(cutmix_index)

        augmented_img, augmented_label = self.cutmix_augmentation(
            origin_image, origin_age_label, cutmix_image, cutmix_age_label
        )
        if self.transform:
            image_transform = self.transform(augmented_img)
            return image_transform, augmented_label

        # 라벨로 one-hot vector 리턴.
        return augmented_img, augmented_label

    def __len__(self):
        """데이터셋의 길이를 반환하는 메서드"""
        return len(self.image_paths)

    def get_age_label(self, index, skewed=False) -> AgeLabels:
        """인덱스에 해당하는 나이 라벨을 반환하는 메서드"""
        if skewed:
            return F.one_hot(
                torch.tensor(self.skewed_age_labels[index] * 1), self.num_classes
            )
        return F.one_hot(torch.tensor(self.age_labels[index] * 1), self.num_classes)

    def read_image(self, index, skewed=False):
        """인덱스에 해당하는 이미지를 읽는 메서드"""
        if skewed:
            image_path = self.skewed_image_paths[index]
        else:
            image_path = self.image_paths[index]
        return Image.open(image_path)

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """데이터셋을 학습과 검증용으로 나누는 메서드
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여 torch.utils.data.Subset 클래스 둘로 나눕니다.
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set

    def calc_statistics(self):
        """데이터셋의 통계치를 계산하는 메서드"""
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine"
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    @staticmethod
    def denormalize_image(image, mean, std):
        """정규화된 이미지를 원래대로 되돌리는 메서드"""
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp


class GenderModelDataset(Dataset):
    """
    성별 데이터셋의 기본 클래스

    image:  PIL.Image
    labels: Long
    ex) (PIL.Image 타입의 어떤 이미지), (해당 이미지에 맞는 class num (0~3 중 1))
    """

    num_classes = 2

    image_paths = []
    gender_labels = []
    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
        one_hot=False,
    ):
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.mean = mean
        self.std = std
        self.transform = None
        self.one_hot = one_hot
        self.setup()  # 데이터셋을 설정
        self.calc_statistics()

    def setup(self):
        """데이터 디렉토리로부터 이미지 경로와 라벨을 설정하는 메서드"""
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if (
                    _file_name not in self._file_names
                ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)

                self.image_paths.append(img_path)
                self.gender_labels.append(gender_label)

    def set_transform(self, transform):
        """변환(transform)을 설정하는 메서드"""
        self.transform = transform

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        gender_label = self.get_gender_label(index)
        image_transform = self.transform(image)
        return image_transform, gender_label

    def __len__(self):
        """데이터셋의 길이를 반환하는 메서드"""
        return len(self.image_paths)

    def get_gender_label(self, index) -> GenderLabels:
        """인덱스에 해당하는 성별 라벨을 반환하는 메서드"""
        if self.one_hot:
            return F.one_hot(
                torch.tensor(self.gender_labels[index] * 1), self.num_classes
            ).float()
        return self.gender_labels[index]

    def read_image(self, index):
        """인덱스에 해당하는 이미지를 읽는 메서드"""
        image_path = self.image_paths[index]
        return Image.open(image_path)

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """데이터셋을 학습과 검증용으로 나누는 메서드
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여 torch.utils.data.Subset 클래스 둘로 나눕니다.
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set

    def calc_statistics(self):
        """데이터셋의 통계치를 계산하는 메서드"""
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine"
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    @staticmethod
    def denormalize_image(image, mean, std):
        """정규화된 이미지를 원래대로 되돌리는 메서드"""
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp


class CutMixGenderModelDataset(Dataset):
    """GenderModel에 사용하기 위한 CutMix를 적용하는 데이터셋 클래스

    Args:
        data_dir (str): 데이터 위치
        mean
        std
        val_ratio
        cutmix_augmentation (CutMixVerticalAugmentation): CutMix 수행하는 Class
        cutmix_prob (float): CutMix를 수행 할 확률

    """

    num_classes = 2

    image_paths = []
    gender_labels = []

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL,
    }

    def __init__(
        self,
        data_dir,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
        cutmix_prob=0.49,
        one_hot=True,
    ):
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.mean = mean
        self.std = std
        self.transform = None
        self.cutmix_augmentation = CutMixVerticalAugmentation(cutmix_prob)
        self.setup()  # 데이터셋을 설정
        self.calc_statistics()

    def setup(self):
        """데이터 디렉토리로부터 이미지 경로와 라벨을 설정하는 메서드"""
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if (
                    _file_name not in self._file_names
                ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(
                    self.data_dir, profile, file_name
                )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)

                self.image_paths.append(img_path)
                self.gender_labels.append(gender_label)

    def set_transform(self, transform):
        """변환(transform)을 설정하는 메서드"""
        self.transform = transform

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터를 가져오는 메서드"""
        # assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        origin_image = self.read_image(index)
        origin_age_label = self.get_gender_label(index)

        cutmix_index = random.randint(0, len(self.image_paths) - 1)
        cutmix_image = self.read_image(cutmix_index)
        cutmix_gender_label = self.get_gender_label(cutmix_index)
        augmented_img, augmented_label = self.cutmix_augmentation(
            origin_image, origin_age_label, cutmix_image, cutmix_gender_label
        )
        if self.transform:
            image_transform = self.transform(augmented_img)
            return image_transform, augmented_label
        # 라벨로 one-hot vector 리턴.
        return augmented_img, augmented_label

    def __len__(self):
        """데이터셋의 길이를 반환하는 메서드"""
        return len(self.image_paths)

    def get_gender_label(self, index) -> GenderLabels:
        """인덱스에 해당하는 성별 라벨을 반환하는 메서드"""
        return F.one_hot(torch.tensor(self.gender_labels[index] * 1), self.num_classes)

    def read_image(self, index):
        """인덱스에 해당하는 이미지를 읽는 메서드"""
        image_path = self.image_paths[index]
        return Image.open(image_path)

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """데이터셋을 학습과 검증용으로 나누는 메서드
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여 torch.utils.data.Subset 클래스 둘로 나눕니다.
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set

    def calc_statistics(self):
        """데이터셋의 통계치를 계산하는 메서드"""
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine"
            )
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image**2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean**2) ** 0.5 / 255

    @staticmethod
    def denormalize_image(image, mean, std):
        """정규화된 이미지를 원래대로 되돌리는 메서드"""
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


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
