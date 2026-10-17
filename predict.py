import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from src.medimage.model import ChestXRayCNN
from src.medimage.utils import load_checkpoint

def predict(image_path, checkpoint_path="artifacts/model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = ChestXRayCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    image = Image.open(image_path).convert("L").resize((28, 28))
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    names = checkpoint.get("class_names", ["normal", "pneumonia"])
    return {names[i]: float(probs[i]) for i in range(len(names))}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a PNG/JPG image")
    parser.add_argument("--checkpoint", default="artifacts/model.pt")
    args = parser.parse_args()
    if not Path(args.checkpoint).exists():
        raise SystemExit("Checkpoint missing. Train the model first.")
    result = predict(args.image, args.checkpoint)
    print("RESEARCH DEMO ONLY — NOT A MEDICAL DIAGNOSIS")
    for label, score in result.items():
        print(f"{label}: {score:.4f}")

if __name__ == "__main__":
    main()
