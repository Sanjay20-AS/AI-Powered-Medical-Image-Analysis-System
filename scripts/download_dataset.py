import sys
from pathlib import Path

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.medimage.data import MedMNISTTorchDataset


def main():
    print("Downloading PneumoniaMNIST using MedMNIST...")

    for split in ("train", "val", "test"):
        dataset = MedMNISTTorchDataset(
            split=split,
            download=True
        )
        print(f"{split}: {len(dataset)} images")

    print("Dataset download completed successfully.")


if __name__ == "__main__":
    main()