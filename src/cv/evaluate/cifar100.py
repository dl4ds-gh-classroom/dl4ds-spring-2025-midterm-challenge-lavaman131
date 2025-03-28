import torch
import torch.nn as nn
from tqdm.auto import tqdm  # For progress bars
from torch.utils.data import DataLoader
from typing import Tuple, List, Union
from omegaconf import DictConfig

from cv.utils.misc import get_dtype


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
