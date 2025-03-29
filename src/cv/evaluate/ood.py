import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import urllib.request
from omegaconf import DictConfig
from typing import List
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import v2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from cv.data.dataset import CIFAR100
from cv.data.transforms import make_classification_eval_transform
from cv.utils.misc import get_dtype


def evaluate_ood(
    config: DictConfig,
    model: nn.Module,
    distortion_name: str,
    severity: int,
) -> List[int]:
    data_dir = config.ood_dir
    device = config.device
    dtype = get_dtype(config.dtype)
    # Load the OOD images
    images = np.load(os.path.join(data_dir, f"{distortion_name}.npy"))

    # Select the subset of images for the given severity
    start_index = (severity - 1) * 10000
    end_index = severity * 10000
    images = images[start_index:end_index]

    # Normalize after converting to tensor
    test_transform = make_classification_eval_transform(
        resize_size=config.transforms.resize_size,
        crop_size=config.input_size,
    )

    dataset = CIFAR100(data=images, transform=test_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        persistent_workers=config.persistent_workers,
    )

    predictions = []  # Store predictions
    with torch.no_grad():
        for inputs in tqdm(
            dataloader,
            desc=f"Evaluating {distortion_name} (Severity {severity})",
            leave=False,
        ):
            with torch.autocast(
                config.device_type, dtype=dtype, enabled=config.use_fp16
            ):
                inputs = inputs.to(device)

                outputs = model(inputs)
                _, predicted = outputs.max(1)

            predictions.extend(predicted.cpu().numpy())
    return predictions


# Check if the files are already downloaded
def files_already_downloaded(directory: str, num_files: int) -> bool:
    for i in range(num_files):
        file_name = f"distortion{i:02d}.npy"
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(file_path):
            return False
    return True


def evaluate_ood_test(config: DictConfig, model: nn.Module) -> List[int]:
    data_dir = config.ood_dir
    device = config.device

    num_files = 19  # Number of files to download

    # Only download if files aren't already downloaded
    if not files_already_downloaded(data_dir, num_files):
        # Create the directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Base URL for the files
        base_url = (
            "https://github.com/DL4DS/ood-test-files/raw/refs/heads/main/ood-test/"
        )

        # Download files distortion00.npy to distortion18.npy
        for i in range(num_files):
            file_name = f"distortion{i:02d}.npy"
            file_url = base_url + file_name
            file_path = os.path.join(data_dir, file_name)

            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(file_url, file_path)
            print(f"Downloaded {file_name} to {file_path}")

        print("All files downloaded successfully.")
    else:
        print("All files are already downloaded.")

    distortions = [f"distortion{str(i).zfill(2)}" for i in range(19)]

    all_predictions = []  # Store all predictions for the submission file

    model.eval()  # Ensure model is in evaluation mode
    for distortion in distortions:
        for severity in range(1, 6):
            predictions = evaluate_ood(
                config=config,
                model=model,
                distortion_name=distortion,
                severity=severity,
            )
            all_predictions.extend(predictions)  # Accumulate predictions
            print(f"{distortion} (Severity {severity})")

    return all_predictions


def create_ood_df(all_predictions: List[int]) -> pd.DataFrame:
    distortions = [f"distortion{str(i).zfill(2)}" for i in range(19)]

    # --- Create Submission File (OOD) ---
    # Create IDs for OOD (assuming the order is as evaluated)
    ids_ood = []
    for distortion in distortions:
        for severity in range(1, 6):
            for i in range(10000):
                ids_ood.append(f"{distortion}_{severity}_{i}")

    submission_df_ood = pd.DataFrame({"id": ids_ood, "label": all_predictions})
    return submission_df_ood
