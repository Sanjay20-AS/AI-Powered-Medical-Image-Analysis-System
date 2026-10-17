import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, ys, preds = [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        ys.extend(labels.detach().cpu().tolist())
        preds.extend(logits.argmax(1).detach().cpu().tolist())
    p, r, f1, _ = precision_recall_fscore_support(
        ys, preds, average="binary", zero_division=0
    )
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "accuracy": float(accuracy_score(ys, preds)),
        "precision_pneumonia": float(p),
        "recall_pneumonia": float(r),
        "f1_pneumonia": float(f1),
        "confusion_matrix": confusion_matrix(ys, preds, labels=[0, 1]).tolist(),
    }
