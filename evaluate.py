from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn
from src.medimage.data import make_loader
from src.medimage.model import ChestXRayCNN
from src.medimage.engine import run_epoch
from src.medimage.utils import load_checkpoint, save_json, ensure_dir

def main():
    checkpoint_path = Path("artifacts/model.pt")
    if not checkpoint_path.exists():
        raise SystemExit("No checkpoint found. Run `python train.py` first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = ChestXRayCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loader = make_loader("test", batch_size=64, shuffle=False, download=True)
    metrics = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
    ensure_dir("artifacts")
    save_json(metrics, "artifacts/evaluation.json")
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["normal", "pneumonia"])
    ax.set_yticks([0, 1], labels=["normal", "pneumonia"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("PneumoniaMNIST test confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig("artifacts/confusion_matrix.png", dpi=160)
    plt.close(fig)
    print("Test metrics (benchmark only):")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
