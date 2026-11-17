
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data import get_eval_transform

# Preprocessing used by the trained model
TRANSFORM = get_eval_transform()

# Same resizing and cropping, but keeps the image suitable for display
DISPLAY = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])


def get_target_layer(model, name):
    """Select the last convolutional block for Grad-CAM."""

    if name == "resnet50":
        return model.layer4[-1]

    if name == "efficientnet_b0":
        return model.features[-1]

    raise ValueError(f"Unknown model name: {name}")


def gradcam_overlay(model, name, image, class_idx, device="cpu"):
    """Generate a heatmap overlay for the selected class."""

    # Convert image to RGB
    image = image.convert("RGB")

    # Prepare image in the format expected by the model
    x = TRANSFORM(image).unsqueeze(0).to(device)

    # Generate Grad-CAM heatmap using the selected target layer
    with GradCAM(
        model=model,
        target_layers=[get_target_layer(model, name)]
    ) as cam:

        heatmap = cam(
            input_tensor=x,
            targets=[ClassifierOutputTarget(class_idx)]
        )[0]

    # Prepare original image as a normalized NumPy array
    rgb = np.array(DISPLAY(image), dtype=np.float32) / 255.0

    # Overlay heatmap on the original image
    return show_cam_on_image(rgb, heatmap, use_rgb=True)
