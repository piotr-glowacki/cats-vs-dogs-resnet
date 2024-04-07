import copy
import argparse
import torch
import torch.onnx
import torch.nn as nn
from typing import Tuple
from utils import Settings


def train_model(settings: Settings) -> Tuple[torch.nn.Module, float]:
    """
    Trains a PyTorch model using the settings provided in Settings class.

    Parameters:
        settings (Settings): A settings object which contains all configurations needed.

    Returns:
        model, best_acc (Tuple[torch.nn.Module, float]): A tuple containing the trained model (torch.nn.Module) and the highest validation accuracy achieved (float).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = settings.model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(settings.num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in settings.dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                settings.optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = settings.criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        settings.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / settings.dataset_sizes[phase]
            epoch_acc = running_corrects.double() / settings.dataset_sizes[phase]

            print(f"{epoch}. {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Best acc (to this point): {best_acc:.4f} \n")

    model.load_state_dict(best_model_wts)
    print(f"Best acc: {best_acc:.4f} \n")
    return model, best_acc


def main(args):
    settings = Settings(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        momentum=args.momentum,
    )
    num_ftrs = settings.model.fc.in_features

    # change last layer to 2 classes & set optimizer
    settings.model.fc = nn.Linear(num_ftrs, len(settings.class_names))
    settings.optimizer = torch.optim.SGD(
        settings.model.parameters(), lr=settings.lr, momentum=settings.momentum
    )

    # start training
    model, best_acc = train_model(settings)
    print(f"Best training accuracy: {best_acc}")

    # save the model
    model_path = "trained_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # export to onnx
    if args.onnx:
        dummy_input = torch.randn(
            1, 3, 224, 224, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        onnx_file_path = "cats_vs_dogs_pytorch_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_file_path,
            export_params=True,
            opset_version=13,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"Model has been converted to ONNX and saved at {onnx_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PyTorch model on the Cats vs Dogs dataset."
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.92, help="Momentum")
    parser.add_argument(
        "--onnx", type=bool, default=True, help="Export to onnx format?"
    )
    args = parser.parse_args()
    main(args)
