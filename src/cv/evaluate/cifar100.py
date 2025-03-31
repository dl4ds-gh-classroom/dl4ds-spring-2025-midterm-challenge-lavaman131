import torch
import torch.nn as nn
from tqdm.auto import tqdm  # For progress bars
from torch.utils.data import DataLoader
from typing import Tuple, List, Union
from omegaconf import DictConfig
from cv.data.transforms import get_shift_transforms
from cv.utils.misc import get_dtype
from torch.nn import functional as F


# --- Evaluation on Clean CIFAR-100 Test Set ---
def evaluate_cifar100_test(
    config: DictConfig,
    model: nn.Module,
    test_loader: DataLoader,
    device: Union[str, torch.device],
) -> Tuple[List[int], float]:
    """Evaluation on clean CIFAR-100 test set."""

    dtype = get_dtype(config.dtype)

    model.eval()
    correct = 0
    total = 0
    predictions = []  # Store predictions for the submission file
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(
            tqdm(test_loader, desc="Evaluating on Clean Test Set")
        ):
            with torch.autocast(
                config.device_type, dtype=dtype, enabled=config.use_fp16
            ):
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)

            predictions.extend(
                predicted.cpu().numpy()
            )  # Move predictions to CPU and convert to numpy
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

    clean_accuracy = 100.0 * correct / total
    # print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
    return predictions, clean_accuracy


# --- Evaluation on Clean CIFAR-100 Test Set with test time transformations ---
def evaluate_cifar100_ttc(
    config: DictConfig,
    model: nn.Module,
    test_loader: DataLoader,
    device: Union[str, torch.device],
) -> Tuple[List[int], float]:
    """Evaluation on clean CIFAR-100 test set."""

    dtype = get_dtype(config.dtype)

    model.eval()
    correct = 0
    total = 0
    predictions = []  # Store predictions for the submission file

    shifts = [i for i in range(1, 3)]
    transforms = get_shift_transforms(shifts, "Moore")

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(
            tqdm(test_loader, desc="Evaluating on Clean Test Set")
        ):
            inputs = inputs.to(device, non_blocking=True)
            probabilities = torch.zeros(
                (inputs.shape[0] * len(transforms), config.num_classes),
                device=device,
                dtype=torch.float32,
            )
            transformed_inputs = torch.cat(
                [transform(inputs) for transform in transforms], dim=0
            )

            with torch.autocast(
                config.device_type, dtype=dtype, enabled=config.use_fp16
            ):
                outputs = model(transformed_inputs)

            probabilities += F.softmax(outputs, dim=-1)

            probabilities = probabilities.view(
                len(transforms), inputs.shape[0], config.num_classes
            ).mean(0)

            predicted = probabilities.argmax(1)

            predictions.extend(
                predicted.cpu().numpy()
            )  # Move predictions to CPU and convert to numpy
            total += labels.size(0)
            correct += predicted.eq(labels.to(device, non_blocking=True)).sum().item()

    clean_accuracy = 100.0 * correct / total
    # print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
    return predictions, clean_accuracy
