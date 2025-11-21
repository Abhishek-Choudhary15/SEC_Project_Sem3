# model.py
import torch
import torch.nn as nn
import torchvision.models as models

def get_model(model_name='resnet18', pretrained=True, num_classes=1):
    """
    Returns a model for binary classification (deepfake vs real).
    num_classes=1 -> use BCEWithLogitsLoss (sigmoid).
    """
    model_name = model_name.lower()
    if model_name == 'resnet18':
        m = models.resnet18(pretrained=pretrained)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        m = models.mobilenet_v2(pretrained=pretrained)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return m
