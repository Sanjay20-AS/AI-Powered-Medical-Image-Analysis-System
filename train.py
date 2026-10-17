import argparse
import csv
from pathlib import Path
import torch
from torch import nn
from torch.optim import AdamW
from src.medimage.data import make_loader
from src.medimage.model import ChestXRayCNN
from src.medimage.engine import run_epoch
from src.medimage.utils import ensure_dir, save_json, set_seed

def main():
    parser = argparse.ArgumentParser(description="Train a PneumoniaMNIST CNN baseline.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader = make_loader("train", args.batch_size, True, args.workers)
    val_loader = make_loader("val", args.batch_size, False, args.workers)
    model = ChestXRayCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    artifacts = ensure_dir("artifacts")
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": train_metrics["loss"],
               "train_accuracy": train_metrics["accuracy"],
               "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"],
               "val_precision_pneumonia": val_metrics["precision_pneumonia"],
               "val_recall_pneumonia": val_metrics["recall_pneumonia"],
               "val_f1_pneumonia": val_metrics["f1_pneumonia"]}
        history.append(row)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in row.items()))
        if val_metrics["f1_pneumonia"] > best_f1:
            best_f1 = val_metrics["f1_pneumonia"]
            torch.save({"model_state_dict": model.state_dict(),
                        "class_names": ["normal", "pneumonia"],
                        "image_size": 28, "seed": args.seed,
                        "validation_metrics": val_metrics}, artifacts / "model.pt")
    with (artifacts / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    save_json({"device": str(device), "epochs": args.epochs, "seed": args.seed,
               "best_validation_f1_pneumonia": best_f1}, artifacts / "metrics.json")
    print(f"Best checkpoint saved to {artifacts / 'model.pt'}")

if __name__ == "__main__":
    main()
