# GitHub proof-of-work guide

## 1. Create the repository

Create a public repository named `AI-Powered-Medical-Image-Analysis-System`. Add a description such as:

> Educational medical computer-vision pipeline using public MedMNIST data, PyTorch, model evaluation, Grad-CAM-style debugging, and Streamlit. Not for clinical use.

## 2. Upload code

From the project directory:

```bash
git init
git add .
git commit -m "Initial medical image analysis pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AI-Powered-Medical-Image-Analysis-System.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. Create the remote repository first and do not commit `data/`, `artifacts/`, `.venv/`, or patient information.

## 3. Build evidence

- Run the dataset download, training, evaluation, and Streamlit app.
- Save a screenshot of the app landing page, evaluation metrics, confusion matrix, and explanation overlay.
- Add a short screen recording showing the pipeline.
- Fill in the experiment log using actual output.
- Include a brief section on failure cases and limitations.
- Link to the MedMNIST paper and dataset page.

## 4. Recommended README additions

Add screenshots under `docs/images/` and reference them with relative paths, e.g. `![Dashboard](docs/images/dashboard.png)`. Only use screenshots you created from your own run.

## 5. Resume bullet template

> Built an educational medical-image classification pipeline using PyTorch and MedMNIST, including reproducible training, held-out evaluation, confusion-matrix reporting, Streamlit inference UI, and Grad-CAM-style visualization; documented dataset limitations and responsible-use boundaries.

Only include technologies and results you can explain confidently. Add metrics only after measuring them yourself.

## 6. Interview talking points

- Why split train, validation, and test data?
- Why accuracy alone can be misleading?
- What do precision and recall mean for the pneumonia class?
- Why is 28×28 resolution a major limitation?
- What is domain shift?
- Why is a heatmap not proof of explainability?
- What additional evidence is needed before clinical deployment?
