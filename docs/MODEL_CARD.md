# Model card

## Model

- Name: Educational PneumoniaMNIST CNN baseline
- Task: Two-class image classification (`normal`, `pneumonia`)
- Input: 1-channel image resized to 28×28 and normalized to [-1, 1]
- Output: Two softmax scores
- Framework: PyTorch
- Intended use: Coursework, portfolio demonstration, model-development practice

## Prohibited / out-of-scope use

Not intended for screening, diagnosis, triage, treatment selection, or any clinical decision. Not validated as a medical device. Do not use on real patients.

## Training data

PneumoniaMNIST via MedMNIST; see `DATASET.md`. The training script uses the official training split for optimization and validation split for checkpoint selection. The test split is reserved for evaluation.

## Evaluation

Run `python evaluate.py`. Store actual results and the command/environment in `docs/EXPERIMENT_LOG.md`. Do not insert fabricated metrics.

## Known risks

- Low resolution and small dataset
- Dataset and demographic bias
- Domain shift between benchmark and real-world images
- Potential shortcut learning
- Uncalibrated scores
- Heatmap explanations can be misleading
- Model is not validated for clinical use

## Human oversight

No clinical workflow should rely on this model. Any real-world medical AI requires appropriate data governance, representative external validation, expert review, risk management, regulatory assessment, and institutional approval.
