from torchvision import datasets
from typing import Callable, Optional, Any, Tuple, Union
from PIL import Image
import numpy as np


class CIFAR100(datasets.CIFAR100):
    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        *args: Any,
        **kwargs: Any,
    ):
        if data is None:
            super().__init__(*args, **kwargs)
        else:
            self.data = data
            self.targets = targets
            self.transform = kwargs.get("transform", None)
            self.target_transform = kwargs.get("target_transform", None)

    def __getitem__(self, index: int) -> Union[Any, Tuple[Any, Any]]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]

        if self.targets is not None:
            target = self.targets[index]
        else:
            target = None

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented["image"]

        if target is None:
            return img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
