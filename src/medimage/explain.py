
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib


def gradcam_overlay(model, tensor, original_image):
    """Generate an educational Grad-CAM overlay."""

    model.eval()
    target_layer = model.features[8]
    activations = {}

    def forward_hook(_module, _inputs, output):
        # Store the actual tensor used by the model's computation graph.
        activations["value"] = output

    hook = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            input_tensor = tensor.detach().clone().requires_grad_(True)
            logits = model(input_tensor)

            activation = activations["value"]
            target_class = logits.argmax(dim=1)
            score = logits[0, target_class.item()]

            gradients = torch.autograd.grad(
                outputs=score,
                inputs=activation,
                retain_graph=False,
                create_graph=False,
            )[0]

            # Detach only after gradients have been calculated.
            weights = gradients.detach().mean(
                dim=(2, 3), keepdim=True
            )
            acts = activation.detach()

            cam = torch.relu(
                (weights * acts).sum(dim=1, keepdim=True)
            )

            cam = F.interpolate(
                cam,
                size=(original_image.height, original_image.width),
                mode="bilinear",
                align_corners=False,
            )

            cam = cam[0, 0].cpu().numpy()
            cam_min, cam_max = cam.min(), cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        heat = np.uint8(255 * cam)
        rgb_heat = np.uint8(
    matplotlib.colormaps["jet"](heat)[..., :3] * 255
    )

        base = original_image.convert("RGB")
        heat_image = Image.fromarray(rgb_heat)

        return Image.blend(base, heat_image, alpha=0.35)

    finally:
        hook.remove()
