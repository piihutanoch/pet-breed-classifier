
import json
from pathlib import Path

import torch

from src.data import get_eval_transform
from src.model import build_model


# Locate the artifacts directory relative to this file
DEFAULT_ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts_v2"

# Initialise the evaluation transformations once for reuse
TRANSFORM = get_eval_transform()


def load_model(artifacts_dir=DEFAULT_ARTIFACTS, device="cpu"):
    """Load the trained model and return it with its class labels."""

    artifacts_dir = Path(artifacts_dir)

    # Load model configuration and index-to-breed mapping
    with open(artifacts_dir / "model_config.json") as f:
        config = json.load(f)

    with open(artifacts_dir / "classes.json") as f:
        idx_to_class = json.load(f)

    # Convert the index-to-class dictionary into an ordered list
    classes = [idx_to_class[str(i)] for i in range(config["num_classes"])]

    # Reconstruct the selected architecture without downloading pretrained weights
    model = build_model(
        config["name"],
        num_classes=config["num_classes"],
        freeze=False,
        pretrained=False
    )

    # Load the trained weights saved during model training
    model.load_state_dict(
        torch.load(artifacts_dir / "model.pt", map_location=device)
    )

    # Move model to the specified device and switch to evaluation mode
    model.to(device)
    model.eval()

    return model, classes, config["name"]


def predict(model, classes, image, k=5, device="cpu"):
    """Return the top-k predicted breeds and their probabilities."""

    # Ensure image has three colour channels
    image = image.convert("RGB")

    # Apply evaluation transformations and add batch dimension
    x = TRANSFORM(image).unsqueeze(0).to(device)

    # Perform inference without calculating gradients
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    # Extract the k highest-probability breed predictions
    top = probs.topk(k)

    # Return breed names paired with their probabilities
    return [
        (classes[i], float(p))
        for p, i in zip(top.values, top.indices)
    ]
