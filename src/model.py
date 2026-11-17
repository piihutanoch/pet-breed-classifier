
import torch.nn as nn
from torchvision import models


def build_model(name="efficientnet_b0", num_classes=37, freeze=True, pretrained=True):
    """
    Build an EfficientNet-B0 or ResNet50 model for breed classification.

    pretrained=True: Load ImageNet pretrained weights.
    pretrained=False: Initialise the backbone with random weights.
    freeze=True: Freeze backbone parameters while keeping the new classifier trainable.
    """

    if name == "efficientnet_b0":
        # Load EfficientNet-B0 with ImageNet weights if requested
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        # Freeze backbone parameters to prevent their updates during training
        if freeze:
            for param in model.parameters():
                param.requires_grad = False

        # Replace the original classification layer with a num_classes output layer
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif name == "resnet50":
        # Load ResNet50 with ImageNet weights if requested
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)

        # Freeze backbone parameters
        if freeze:
            for param in model.parameters():
                param.requires_grad = False

        # Replace the original fully connected layer with a num_classes output layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        # Raise an error if an unsupported architecture is requested
        raise ValueError(f"Unknown model name: {name}")

    return model


# Identify the classifier layer name for each architecture
HEAD_PREFIX = {
    "efficientnet_b0": "classifier.",
    "resnet50": "fc."
}


def split_params(model, name):
    """
    Separate model parameters into backbone and classifier head groups.
    This allows different learning rates to be assigned to each group.
    """

    backbone, head = [], []

    # Iterate through all named model parameters
    for param_name, param in model.named_parameters():

        # Identify classifier head parameters using the architecture-specific prefix
        if param_name.startswith(HEAD_PREFIX[name]):
            head.append(param)
        else:
            backbone.append(param)

    return backbone, head
