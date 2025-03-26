from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from tqdm import tqdm
from typing import Tuple
import torch


def train_one_epoch(
    config: DictConfig,
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    """Train one epoch, e.g. all batches of one epoch."""
    device = config.device
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can printprogress
    progress_bar = tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Train]", leave=False
    )

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):
        # move inputs and labels to the target device
        inputs, labels = (
            inputs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        optimizer.zero_grad()

        ### TODO - Your code here
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.detach().cpu().item()
        _, predicted = y_pred.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix(
            {"loss": running_loss / (i + 1), "acc": 100.0 * correct / total}
        )

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    return train_loss, train_acc


################################################################################
# Define a validation function
################################################################################
def validate(
    config: DictConfig,
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    """Validate the model"""
    model.eval()  # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track gradients
        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(val_loader, desc="[Validate]", leave=False)

        # Iterate throught the validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            # move inputs and labels to the target device
            inputs, labels = (
                inputs.to(config.device, non_blocking=True),
                labels.to(config.device, non_blocking=True),
            )

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            running_loss += loss.detach().cpu().item()
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix(
                {"loss": running_loss / (i + 1), "acc": 100.0 * correct / total}
            )

    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc
