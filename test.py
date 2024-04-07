import torch
import torch.nn as nn
from torchvision import models
from timeit import default_timer as timer
from utils import Settings
from typing import Tuple


def test_model(model: nn.Module, criterion: nn.Module) -> float:
    """
    Evaluates a PyTorch model on the test dataset using the specified criterion.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        criterion (torch.nn.modules.loss._Loss): The loss function used for evaluation.

    Returns:
        test_acc (float): The accuracy of the model on the test dataset as a percentage.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in settings.dataloaders["test"]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    # test_loss = running_loss / len(data_utils.dataloaders["test"].dataset)
    test_acc = running_corrects.double() / len(settings.dataloaders["test"].dataset)

    return test_acc * 100


def measure_inference_time(model: nn.Module, iterations: int = 300) -> float:
    """
    Measures the average inference time of the model using a dummy input.

    Parameters:
        model (nn.Module): The model to be evaluated.
        iterations (int): Number of iterations to measure the inference time.

    Returns:
        float: Average inference time per input in seconds.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_input = torch.randn(
        1, 3, 224, 224, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    total_time = 0.0
    for i in range(iterations):
        start_time = timer()
        with torch.no_grad():
            model(dummy_input)
        end_time = timer()
        total_time += end_time - start_time

    average_time = total_time / iterations
    return average_time


if __name__ == "__main__":
    settings = Settings()

    # load the model
    model_path = "trained_model.pth"
    model = models.resnet18(pretrained=False)
    num_ftrs = settings.model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(settings.class_names))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # calculate accuracy
    test_acc = test_model(model, settings.criterion)
    print(f"Accuracy on test set: {test_acc:.2f}%")

    # measure time
    inference_time = measure_inference_time(model)
    print(f"Inference time: {inference_time:.6f}% seconds per iteration")
