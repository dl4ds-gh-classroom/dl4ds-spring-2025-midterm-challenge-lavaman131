import torch
import torch.nn as nn
from tqdm.auto import tqdm  # For progress bars
from torch.utils.data import DataLoader
from typing import Tuple, List, Union


# --- Evaluation on Clean CIFAR-100 Test Set ---
def evaluate_cifar100_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: Union[str, torch.device],
) -> Tuple[List[int], float]:
    """Evaluation on clean CIFAR-100 test set."""

    model.eval()
    correct = 0
    total = 0
    predictions = []  # Store predictions for the submission file
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(
            tqdm(test_loader, desc="Evaluating on Clean Test Set")
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
