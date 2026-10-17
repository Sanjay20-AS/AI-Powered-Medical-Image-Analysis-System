import torch
from src.medimage.model import ChestXRayCNN

def test_model_output_shape():
    model = ChestXRayCNN()
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, 2)

def test_model_outputs_finite():
    model = ChestXRayCNN()
    y = model(torch.randn(2, 1, 28, 28))
    assert torch.isfinite(y).all()
