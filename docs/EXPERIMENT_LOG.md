# Experiment log

Fill this in with real outputs after each run.

| Date | Commit | Dataset/version | Seed | Epochs | Batch size | Best validation F1 | Test accuracy | Notes |
|---|---|---|---:|---:|---:|---:|---:|---|
| TODO | TODO | PneumoniaMNIST / MedMNIST version | 42 | 5 | 64 | TODO | TODO | Run locally; do not fabricate values |

## Commands

```bash
python scripts/download_dataset.py
python train.py --epochs 5 --batch-size 64 --seed 42
python evaluate.py
```

## Environment

- OS: TODO
- Python: TODO
- PyTorch: TODO
- CPU/GPU: TODO
- Dataset package version: TODO

## Interpretation

Report precision, recall, F1, confusion matrix, and class counts—not accuracy alone. State that benchmark metrics are not clinical evidence.
