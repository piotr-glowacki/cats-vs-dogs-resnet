import os
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Callable, Tuple, List


class Settings:
    class_names: List[str] = ["Cat", "Dog"]
    DATA_DIR: str = "PetImages"
    train_dir: str = os.path.join(DATA_DIR, "train")
    val_dir: str = os.path.join(DATA_DIR, "val")
    test_dir: str = os.path.join(DATA_DIR, "test")

    data_transforms: Dict[str, Callable] = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    def __init__(
        self,
        lr: float = 0.001,
        momentum: float = 0.92,
        epochs: int = 50,
        batch_size: int = 16,
        num_workers: int = 4,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        model: nn.Module = models.resnet18(weights=ResNet18_Weights.DEFAULT),
    ) -> None:
        self.lr: float = lr
        self.momentum: float = momentum
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.criterion: nn.Module = criterion
        self.model: nn.Module = model

        self.image_datasets: Dict[str, Dataset] = {
            "train": datasets.ImageFolder(
                self.train_dir, self.data_transforms["train"]
            ),
            "val": datasets.ImageFolder(self.val_dir, self.data_transforms["val"]),
            "test": datasets.ImageFolder(self.test_dir, self.data_transforms["test"]),
        }

        self.dataset_sizes: Dict[str, int] = {
            "train": len(self.image_datasets["train"]),
            "val": len(self.image_datasets["val"]),
            "test": len(self.image_datasets["test"]),
        }

        self.dataloaders: Dict[str, DataLoader] = {
            "train": DataLoader(
                self.image_datasets["train"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            ),
            "val": DataLoader(
                self.image_datasets["val"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            ),
            "test": DataLoader(
                self.image_datasets["test"],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            ),
        }
