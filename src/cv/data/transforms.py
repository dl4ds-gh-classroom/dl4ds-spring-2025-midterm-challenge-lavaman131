from functools import partial
from torchvision.transforms import v2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from typing import Sequence, Union, Tuple, Literal, List, Callable
import torch
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from cv.utils.misc import make_2tuple


def make_normalize_transform(
    *,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> A.Compose:
    return A.Compose(
        [
            A.CLAHE(p=1.0, tile_grid_size=(4, 4)),
            A.Normalize(p=1.0, mean=mean, std=std, normalization="standard"),
        ]
    )


# from: https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/transforms.py
def make_classification_train_transform(
    *,
    transform_config: DictConfig,
    crop_size: Union[int, Tuple[int, int]] = 224,
    interpolation: v2.InterpolationMode = cv2.INTER_CUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> A.Compose:
    crop_size = make_2tuple(crop_size)
    transforms_list = [
        A.Pad(p=1.0, padding=4),
        A.RandomResizedCrop(crop_size, scale=(0.32, 1.0), interpolation=interpolation),
        A.HorizontalFlip(p=transform_config.hflip_prob),
        A.RandomBrightnessContrast(p=transform_config.brightness_prob),
        A.GaussNoise(
            p=transform_config.gaussian_noise_prob,
            mean_range=(0.0, 0.0),
            std_range=(0.025, 0.1),
        ),
        A.Rotate(
            p=transform_config.rotation_prob,
            limit=(-15, 15),
            interpolation=interpolation,
        ),
        A.ColorJitter(
            p=transform_config.color_jitter_prob,
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
        ),
        A.ImageCompression(
            p=transform_config.jpeg_compression_prob, quality_range=(90, 99)
        ),
    ]

    transforms_list.extend(
        [
            *make_normalize_transform(mean=mean, std=std),
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
        *make_normalize_transform(mean=mean, std=std),
        ToTensorV2(),
    ]  # type: ignore
    return A.Compose(transforms_list)


# ==================== MODEL TRANSFORMS ====================


def compute_shift_directions(
    pattern: Literal["Neumann", "Moore"],
) -> List[Tuple[int, int]]:
    # Precompute neighbourhood shift unit vectors
    shifts = [  # shifts in yx format
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    shift_directions: List[Tuple[int, int]] = []
    for i in range(len(shifts)):
        if pattern == "Neumann" and i % 2 == 1:
            shift_directions.append(shifts[i])
        elif pattern == "Moore" and i != 4:
            shift_directions.append(shifts[i])
    return shift_directions


def get_shift_transforms(
    dists: List[int], pattern: Literal["Neumann", "Moore"]
) -> List[Callable]:
    transforms: List[Callable] = []
    shifts = compute_shift_directions(pattern)

    def roll_arg_rev(shift: Tuple[int, int], x: torch.Tensor) -> torch.Tensor:
        return torch.roll(x, shift, dims=(-2, -1))

    for d in dists:
        for s in shifts:
            shift = (d * s[0], d * s[1])
            transform = partial(roll_arg_rev, shift)
            transforms.append(transform)
    return transforms
