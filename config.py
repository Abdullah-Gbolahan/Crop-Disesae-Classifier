import os
import torch

# Model Configuration
MODEL_PATH = 'model/crop_disease_model.pth'
CLASS_NAMES_PATH = 'model/class_names.json'
DISEASE_INFO_PATH = 'assets/disease_info.json'

# PyTorch Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image Processing
IMG_SIZE = (224, 224)  # EfficientNetB0 input size
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

# Model Metadata
MODEL_NAME = "Crop Disease Classifier"
MODEL_VERSION = "1.0.0"
MODEL_ARCHITECTURE = "EfficientNetB0 (PyTorch)"

# App Configuration
APP_TITLE = "🌱 AI-Powered Crop Disease Classifier"
APP_ICON = "🌱"
CONFIDENCE_THRESHOLD = 0.75

# ImageNet normalization values for EfficientNet
IMAGENET_MEAN = [0.4728, 0.5311, 0.3941]
IMAGENET_STD = [0.2185, 0.2149, 0.2605]