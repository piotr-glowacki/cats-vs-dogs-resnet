# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from utils import Settings as utils_settings


# class Settings:
#     data_transforms = {
#         "train": transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]
#         ),
#         "val": transforms.Compose(
#             [
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]
#         ),
#         "test": transforms.Compose(
#             [
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]
#         ),
#     }

#     def __init__(self, batch_size: int = 16, num_workers: int = 4) -> None:
#         self.batch_size = batch_size
#         self.num_workers = num_workers

#         self.image_datasets = {
#             "train": datasets.ImageFolder(
#                 utils_settings.train_dir, self.data_transforms["train"]
#             ),
#             "val": datasets.ImageFolder(
#                 utils_settings.val_dir, self.data_transforms["val"]
#             ),
#             "test": datasets.ImageFolder(
#                 utils_settings.test_dir, self.data_transforms["test"]
#             ),
#         }

#         self.dataset_sizes = {
#             "train": len(self.image_datasets["train"]),
#             "val": len(self.image_datasets["val"]),
#             "test": len(self.image_datasets["test"]),
#         }

#         self.dataloaders = {
#             "train": DataLoader(
#                 self.image_datasets["train"],
#                 batch_size=self.batch_size,
#                 shuffle=True,
#                 num_workers=self.num_workers,
#             ),
#             "val": DataLoader(
#                 self.image_datasets["val"],
#                 batch_size=self.batch_size,
#                 shuffle=True,
#                 num_workers=self.num_workers,
#             ),
#             "test": DataLoader(
#                 self.image_datasets["test"],
#                 batch_size=self.batch_size,
#                 shuffle=True,
#                 num_workers=self.num_workers,
#             ),
#         }
