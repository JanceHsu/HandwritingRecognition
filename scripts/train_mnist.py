"""Train a CUDA-friendly MNIST classifier and export a TorchScript model.

Default behavior keeps the original CPU-compatible export path, but the
training loop now prefers CUDA when available, uses a small CNN, mixed
precision on GPU, and a better data pipeline for higher accuracy.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def train(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer, device, scaler=None):
    model.train()
    size = len(dataloader.dataset)
    for batch, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if device.type == "cuda":
            inputs = inputs.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if batch % 100 == 0:
            current = batch * len(inputs)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader: DataLoader, model: nn.Module, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if device.type == "cuda":
                inputs = inputs.to(memory_format=torch.channels_last)
            predictions = model(inputs)
            test_loss += loss_fn(predictions, targets).item()
            correct += (predictions.argmax(1) == targets).type(torch.float).sum().item()

    test_loss /= max(len(dataloader), 1)
    correct /= size
    print(f"Test Error:\n Accuracy: {100 * correct:>0.1f}%, Avg loss: {test_loss:>8f}\n")
    return correct


def build_dataloaders(data_root: Path, batch_size: int, mirror_url: str):
    datasets.MNIST.mirrors = [mirror_url]
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.10, 0.10),
            scale=(0.90, 1.10),
            shear=5,
        ),
        transforms.ToTensor(),
    ])
    test_transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root=str(data_root), train=True, transform=train_transform, download=True)
    mnist_test = datasets.MNIST(root=str(data_root), train=False, transform=test_transform, download=True)

    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size)
    return train_loader, test_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MNIST MLP and export TorchScript")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--mirror-url",
        type=str,
        default="https://storage.googleapis.com/cvdf-datasets/mnist/",
        help="MNIST download mirror",
    )
    parser.add_argument("--device", type=str, default=None, help="device to use: cpu|cuda|auto (auto prefers cuda if available)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader num_workers")
    parser.add_argument("--pin-memory", action="store_true", help="enable DataLoader pin_memory when using CUDA")
    parser.add_argument("--no-augment", action="store_true", help="disable MNIST augmentation")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Resolve device: prefer explicit arg, then CUDA if available when 'auto' or None
    requested = (args.device or "auto").lower()
    if requested == "cpu":
        device = torch.device("cpu")
    elif requested == "cuda":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Using device: {device}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")

    train_loader, test_loader = build_dataloaders(
        args.data_root,
        args.batch_size,
        args.mirror_url,
    )

    if args.no_augment:
        datasets.MNIST.mirrors = [args.mirror_url]
        plain_train = datasets.MNIST(root=str(args.data_root), train=True, transform=transforms.ToTensor(), download=True)
        train_loader = DataLoader(plain_train, batch_size=args.batch_size, shuffle=True)

    # Rebuild dataloaders with performance flags when using CUDA
    dl_kwargs = {}
    if device.type == "cuda":
        safe_num_workers = max(0, args.num_workers)
        if os.name == "nt":
            safe_num_workers = min(safe_num_workers, 2)
        dl_kwargs = {
            "num_workers": safe_num_workers,
            "pin_memory": args.pin_memory,
            "persistent_workers": safe_num_workers > 0,
        }
    train_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    test_loader = DataLoader(
        test_loader.dataset,
        batch_size=args.batch_size,
        **(
            {
                "num_workers": dl_kwargs.get("num_workers", 0),
                "persistent_workers": dl_kwargs.get("persistent_workers", False),
            }
            if device.type == "cuda"
            else {}
        ),
    )

    model = NeuralNetwork().to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}\n-------------")
        train(train_loader, model, loss_fn, optimizer, device, scaler)
        accuracy = test(test_loader, model, loss_fn, device)
        print(f"epoch={epoch + 1}, accuracy={accuracy:.4f}")

    model_path = args.output_dir / "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch Model State to {model_path}")

    # For C++ compatibility, export a CPU-traced TorchScript model even when trained on CUDA.
    model = model.to("cpu")
    example = torch.rand(1, 1, 28, 28)
    traced = torch.jit.trace(model.cpu(), example)
    traced_path = args.output_dir / "mnist_model.pt"
    traced.save(str(traced_path))
    print(f"Saved TorchScript model to {traced_path} (CPU-traced for libtorch compatibility)")


if __name__ == "__main__":
    main()