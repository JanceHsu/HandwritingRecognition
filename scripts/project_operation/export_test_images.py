"""Export sample MNIST images for C++ inference checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets, transforms


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a few MNIST test images")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("test_images"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--mirror-url",
        type=str,
        default="https://storage.googleapis.com/cvdf-datasets/mnist/",
        help="MNIST download mirror",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets.MNIST.mirrors = [args.mirror_url]
    dataset = datasets.MNIST(root=str(args.data_root), train=False, download=True, transform=transforms.ToTensor())

    for index in range(min(args.count, len(dataset))):
        image, label = dataset[index]
        image_path = args.output_dir / f"mnist_{index:02d}_label_{label}.png"
        transforms.ToPILImage()(image).save(image_path)
        print(f"saved {image_path}")


if __name__ == "__main__":
    main()