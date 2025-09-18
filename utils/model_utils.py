import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import numpy as np
from PIL import Image
import streamlit as st
from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

class ConvertToRGB(object):
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

# Define image preprocessing transforms
def get_transforms():
    """Get image preprocessing transforms for EfficientNetB0"""
    return transforms.Compose([
        ConvertToRGB(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def create_model(num_classes):
    """Create EfficientNetB0 model with custom number of classes"""
    model = efficientnet_b0(pretrained=False)
    
    # Replace the classifier layer
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model

def load_pytorch_model(model_path, num_classes, device):
    """Load PyTorch model from checkpoint"""
    try:
        # Create model architecture
        model = create_model(num_classes)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        return model
    
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def preprocess_image_pytorch(image, transform):
    """Preprocess image for PyTorch model"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms and add batch dimension
        processed_image = transform(image).unsqueeze(0)
        
        return processed_image
    
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def get_prediction_confidence_color(confidence):
    """Return color based on confidence level"""
    if confidence >= 0.9:
        return "green"
    elif confidence >= 0.75:
        return "orange"
    else:
        return "red"

def format_confidence(confidence):
    """Format confidence as percentage"""
    return f"{confidence:.1%}"