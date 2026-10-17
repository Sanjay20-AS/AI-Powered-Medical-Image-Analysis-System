from pathlib import Path
import random
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from src.medimage.model import ChestXRayCNN
from src.medimage.explain import gradcam_overlay
from src.medimage.utils import load_checkpoint

st.set_page_config(page_title="Medical Image Analysis Lab", page_icon="🩻", layout="wide")
st.title("🩻 AI-Powered Medical Image Analysis System")
st.caption("Educational computer-vision prototype • PneumoniaMNIST benchmark")

st.warning(
    "**Research/education only. Not a medical device.** Do not use this tool for diagnosis, "
    "triage, treatment, or decisions about a real patient. If you have a health concern, "
    "consult a qualified healthcare professional."
)

with st.sidebar:
    st.header("Project controls")
    mode = st.radio("Mode", ["Model inference", "Virtual demo"])
    st.markdown("---")
    st.markdown("**Model:** compact CNN · 28×28 grayscale")
    st.markdown("**Classes:** normal / pneumonia")
    st.markdown("**Dataset:** PneumoniaMNIST (MedMNIST)")
    st.caption("The model is a learning baseline, not clinically validated.")

checkpoint_path = Path("artifacts/model.pt")

if mode == "Virtual demo":
    st.subheader("Virtual demo mode")
    st.info("This mode generates a random illustrative result. It does not analyze the uploaded image and is not a model prediction.")
    demo_image = st.file_uploader("Optional image for interface preview", type=["png", "jpg", "jpeg"])
    if demo_image:
        st.image(Image.open(demo_image), caption="Preview only — not analyzed", width=350)
    if st.button("Generate simulated result"):
        value = random.uniform(0.55, 0.95)
        st.metric("Simulated pneumonia score", f"{value:.1%}")
        st.progress(value)
        st.write("**Simulated label:**", "pneumonia-like" if value >= 0.70 else "normal-like")
        st.caption("Random simulation only. Do not interpret as a finding about an image or person.")
else:
    st.subheader("Run benchmark model inference")
    if not checkpoint_path.exists():
        st.error("No trained model found at `artifacts/model.pt`. Train it with `python train.py --epochs 5`, or switch to Virtual demo mode.")
    else:
        uploaded = st.file_uploader("Upload a PNG/JPG image", type=["png", "jpg", "jpeg"])
        if uploaded:
            original = Image.open(uploaded).convert("L")
            st.image(original, caption="Uploaded image (input)", width=350)
            st.warning("Only intended for experimentation with compatible benchmark-style images. Arbitrary uploads are out of distribution.")
            if st.button("Analyze image"):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = load_checkpoint(checkpoint_path, device)
                model = ChestXRayCNN().to(device)
                model.load_state_dict(checkpoint["model_state_dict"])
                transform = transforms.Compose([
                    transforms.Resize((28, 28)), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
                tensor = transform(original).unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
                names = checkpoint.get("class_names", ["normal", "pneumonia"])
                cols = st.columns(2)
                for i, col in enumerate(cols):
                    col.metric(f"{names[i].title()} score", f"{probs[i]:.1%}")
                st.caption("Scores are uncalibrated model outputs, not clinical probabilities or diagnoses.")
                try:
                    overlay = gradcam_overlay(model, tensor, original)
                    st.image(overlay, caption="Grad-CAM-style visualization (debugging aid only)", width=350)
                except Exception as exc:
                    st.info(f"Could not render explanation overlay: {exc}")
                st.json({"class_scores": dict(zip(names, [round(float(x), 5) for x in probs])),
                         "disclaimer": "Educational only; not for clinical use."})

st.markdown("---")
st.subheader("Pipeline")
st.code("Public dataset → preprocessing → CNN training → held-out benchmark evaluation → inference → explanation overlay", language="text")
with st.expander("Limitations and responsible use"):
    st.markdown("""
    - No hospital/PACS integration, DICOM workflow, patient data, or clinical validation.
    - PneumoniaMNIST images are 28×28; meaningful image detail is lost.
    - Dataset bias, label noise, population shift, and acquisition differences can affect results.
    - Heatmaps do not prove clinical reasoning or causal feature use.
    - Never use this output to make medical decisions.
    """)
