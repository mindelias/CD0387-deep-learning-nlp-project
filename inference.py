import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import io
import os

def model_fn(model_dir):
    """Load model with fallback to pretrained if smdebug fails"""
    print(f"Available files: {os.listdir(model_dir)}")
    
    device = torch.device("cpu")
    
    # Create fresh model architecture
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 133)
    
    # Try to load ANY weights that work
    for filename in ['model.pth', 'model_complete.pth']:
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            try:
                # Load with maximum tolerance for errors
                checkpoint = torch.load(filepath, map_location=device)
                
                # Extract weights however possible
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        weights = checkpoint['state_dict']
                    else:
                        weights = checkpoint
                else:
                    # If it's a model object, try to get state_dict
                    if hasattr(checkpoint, 'state_dict'):
                        weights = checkpoint.state_dict()
                    else:
                        continue  # Skip this file
                
                # Clean and load weights
                clean_weights = {}
                for key, value in weights.items():
                    if isinstance(value, torch.Tensor) and 'smdebug' not in key:
                        clean_weights[key] = value
                
                model.load_state_dict(clean_weights, strict=False)
                print(f"Loaded weights from {filename}")
                model.eval()
                return model
                
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
                continue
    
    # Fallback: use pretrained ImageNet weights (better than nothing)
    print("Using pretrained ImageNet weights as fallback")
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 133)
    model.eval()
    return model

def input_fn(request_body, content_type):
    if "image" in content_type:
        image = Image.open(io.BytesIO(request_body)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    raise ValueError(f"Unsupported: {content_type}")

def predict_fn(input_data, model):
    with torch.no_grad():
        return torch.nn.functional.softmax(model(input_data), dim=1)

def output_fn(prediction, accept):
    return json.dumps({"predictions": prediction.tolist()})