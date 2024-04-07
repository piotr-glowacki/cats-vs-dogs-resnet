import os
import shutil
from typing import List
from sklearn.model_selection import train_test_split
from utils import Settings

settings = Settings()


def train_test_val_splitter(class_names: List[str]) -> None:
    """
    Splits image data into training, validation, and testing sets for each category.

    This function organizes images from each category into respective training,
    validation, and testing folders based on predefined ratios. It uses the scikit-learn
    `train_test_split` function to randomly split the data.

    Args:
        class_names(List[str]): Class names as a list.

    Returns:
        None
    """
    for category in class_names:
        category_folder = os.path.join(settings.DATA_DIR, category)
        train_category_folder = os.path.join(settings.train_dir, category)
        val_category_folder = os.path.join(settings.val_dir, category)
        test_category_folder = os.path.join(settings.test_dir, category)

        # Create all the appropriate subfolders
        for folder in [
            train_category_folder,
            val_category_folder,
            test_category_folder,
        ]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        files = os.listdir(category_folder)
        files = [
            file for file in files if file.endswith(".jpg")
        ]  # ignore files other than images
        # Split the data into train and test set in 80/20 proportion
        train_files, test_files = train_test_split(
            files, test_size=0.2, random_state=42
        )
        # Split the train data into train and val set in 75/25 proportion
        train_files, val_files = train_test_split(
            train_files, test_size=0.25, random_state=42
        )

        move_files = lambda files, dst_folder: [
            shutil.move(
                os.path.join(category_folder, file), os.path.join(dst_folder, file)
            )
            for file in files
        ]

        move_files(train_files, train_category_folder)
        move_files(val_files, val_category_folder)
        move_files(test_files, test_category_folder)

        os.rmdir(category_folder)


if __name__ == "__main__":
    train_test_val_splitter(settings.class_names)
