# AI-Powered Medical Image Analysis System

A student-friendly, portfolio-grade **research prototype** for binary chest X-ray image classification (PneumoniaMNIST: `normal` vs `pneumonia`). It includes dataset download, training, evaluation, inference, Grad-CAM-style visual explanations, a Streamlit interface, tests, Docker support, and GitHub-ready documentation.

> **Important safety note:** This is an educational project, not a medical device. It must not be used to diagnose, triage, or make treatment decisions. Dataset performance does not imply clinical performance. The included virtual demo is simulated and does not analyze a real medical image.

## What this project demonstrates

- Public-dataset workflow using [MedMNIST PneumoniaMNIST](https://medmnist.com/)
- Reproducible PyTorch training and evaluation
- Patient-independent clinical validation is **not** claimed; this small benchmark has limitations
- Image preprocessing, probability outputs, confusion matrix, classification report
- Grad-CAM-like heatmap overlay for model debugging (not a clinical explanation)
- Streamlit UI with a clear research-only disclaimer
- Virtual demo mode for exploring the interface without model weights
- Tests, configuration, logging, Dockerfile, and GitHub Actions

## Quick start (Windows PowerShell)

Requires Python 3.10 or 3.11 (recommended).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\download_dataset.py
python train.py --epochs 5
python evaluate.py
streamlit run app.py
```

If PowerShell blocks activation, use `.\.venv\Scripts\Activate.ps1` after allowing scripts for the current user, or run commands from Command Prompt using `.venv\Scripts\activate.bat`.

### Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_dataset.py
python train.py --epochs 5
python evaluate.py
streamlit run app.py
```

The first dataset download requires an internet connection. Training time depends on hardware. CPU training works but may be slower.

## Run the app

```bash
streamlit run app.py
```

- If `artifacts/model.pt` exists, the app offers model inference on uploaded images.
- If no trained checkpoint exists, you can use **Virtual demo** to explore the UI. Virtual demo results are randomly simulated and explicitly labelled as such.
- Uploading an image to the real inference path requires a trained checkpoint. Use PNG/JPG files. The benchmark model expects grayscale chest X-ray-like images; other images are out of distribution.

## Project structure

```text
AI-Powered-Medical-Image-Analysis-System/
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .env.example
├── config/
│   └── default.yaml
├── src/medimage/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── engine.py
│   ├── explain.py
│   └── utils.py
├── scripts/
│   └── download_dataset.py
├── tests/
│   ├── test_model.py
│   └── test_utils.py
├── docs/
│   ├── DATASET.md
│   ├── MODEL_CARD.md
│   ├── EXPERIMENT_LOG.md
│   ├── GITHUB_GUIDE.md
│   └── PROJECT_REPORT_TEMPLATE.md
└── .github/workflows/
    └── tests.yml
```

Dataset files and trained weights are intentionally not bundled in this ZIP. Download the public dataset and create your own model checkpoint locally. Do not commit large datasets, patient data, or model weights without checking licensing and repository policy.

## Dataset

The project uses **PneumoniaMNIST** from MedMNIST, derived from pediatric chest X-ray images and resized to 28×28 pixels. It is useful for learning a pipeline, but is far too small and simplified to establish real-world clinical usefulness. Read `docs/DATASET.md` before training.

The download script fetches the dataset through the `medmnist` package. Dataset terms and source acknowledgements remain applicable.

## Training, evaluation, and prediction

```bash
python train.py --epochs 5 --batch-size 64 --seed 42
python evaluate.py
python predict.py --image path/to/xray.png
```

Training writes:
- `artifacts/model.pt`
- `artifacts/metrics.json`
- `artifacts/training_history.csv`

Evaluation writes:
- `artifacts/evaluation.json`
- `artifacts/confusion_matrix.png`

Do not compare metrics from this dataset directly with hospital systems or published clinical studies without matching data splits and methodology.

## Reproducibility

- Set `--seed` to control random seeds.
- The dataset's official train/validation/test splits are used.
- Test data is not used for gradient updates.
- The best validation checkpoint is retained.
- Results vary by hardware, library versions, and training settings.

## GitHub proof-of-work checklist

1. Create a public repository named `AI-Powered-Medical-Image-Analysis-System`.
2. Upload the source, docs, tests, and a small screenshot; exclude dataset and checkpoints.
3. Run the app locally and capture screenshots of the UI, metrics, and explanation overlay.
4. Commit your actual evaluation output only after running it. Never invent scores.
5. Add a short demo video/GIF, architecture diagram, and experiment log.
6. Open an issue for a future improvement (e.g. calibration, external validation, DICOM support in a safe sandbox).
7. Add a clear README disclaimer and dataset citations.
8. Pin the repository to your GitHub profile and link it from your resume.

See `docs/GITHUB_GUIDE.md`.

## Limitations and responsible use

- Educational benchmark only; not for clinical use.
- No hospital integration, DICOM/PACS connectivity, patient records, or clinical workflow.
- 28×28 images discard substantial diagnostic detail.
- Dataset size, provenance, demographic representation, labels, and domain shift limit generalization.
- A heatmap is not proof that a model used medically meaningful evidence.
- Output probabilities are not calibrated clinical risk estimates.
- Always follow dataset terms and applicable privacy rules.

## License

The source code in this repository is provided under the MIT License (see `LICENSE`). This does not relicense the dataset or third-party dependencies.
