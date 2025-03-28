from torchvision.transforms import v2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from typing import Sequence, Union, Tuple
import torch
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from cv.utils.misc import make_2tuple


# from: https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/transforms.py
def make_classification_train_transform(
    *,
    crop_size: Union[int, Tuple[int, int]] = 224,
    interpolation: v2.InterpolationMode = cv2.INTER_CUBIC,
    hflip_prob: float = 0.5,
    brightness_prob: float = 0.5,
    solarize_prob: float = 0.2,
    color_jitter_prob: float = 0.8,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> A.Compose:
    crop_size = make_2tuple(crop_size)
    transforms_list = [
        A.RandomResizedCrop(crop_size, interpolation=interpolation),
        A.HorizontalFlip(p=hflip_prob),
        A.RandomBrightnessContrast(p=brightness_prob),
        A.Solarize(p=solarize_prob),
        A.ColorJitter(
            p=color_jitter_prob, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
        ),
    ]

    transforms_list.extend(
        [
            A.CLAHE(p=1.0, tile_grid_size=(4, 4)),
            A.Normalize(mean=mean, std=std, normalization="standard"),
            ToTensorV2(),
        ]  # type: ignore
    )
    return A.Compose(transforms_list)


# from: https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/transforms.py
def make_classification_eval_transform(
    *,
    resize_size: Union[int, Tuple[int, int]] = 256,
    interpolation: v2.InterpolationMode = cv2.INTER_CUBIC,
    crop_size: Union[int, Tuple[int, int]] = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> A.Compose:
    crop_size = make_2tuple(crop_size)
    resize_size = make_2tuple(resize_size)
    transforms_list = [
        A.Resize(
            height=resize_size[0], width=resize_size[1], interpolation=interpolation
        ),
        A.CenterCrop(height=crop_size[0], width=crop_size[1]),
        A.CLAHE(p=1.0, tile_grid_size=(4, 4)),
        A.Normalize(mean=mean, std=std, normalization="standard"),
        ToTensorV2(),
    ]  # type: ignore
    return A.Compose(transforms_list)
