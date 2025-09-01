import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageFile


import argparse
import os
import sys
import logging

# SageMaker Debugging and Profiling imports
import smdebug.pytorch as smd
from smdebug import modes
from smdebug.profiler.utils import str2bool


# CRITICAL FIX: Enable loading of truncated/corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, hook):
    '''
    Test function with debugging hooks
    The hook will capture information about what's happening during testing
    '''
    print("Testing Model on Whole Testing Dataset")
    
    # Tell the hook we're in evaluation mode
    hook.set_mode(modes.EVAL)
    
    model.eval()
    running_loss = 0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
    
    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    
    print(f"Testing Loss: {total_loss}")
    print(f"Testing Accuracy: {total_acc}")
    
    return total_acc

def train(model, train_loader, criterion, optimizer, epochs, hook):
    '''
    Training function with debugging hooks
    The hook captures training metrics like loss, gradients, and weights
    '''
    print("Training Model...")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Tell the hook we're in training mode
        hook.set_mode(modes.TRAIN)
        model.train()
        
        running_loss = 0.0
        running_corrects = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            
            # Log progress every few batches
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        
        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    return model

def net(num_classes):
    '''
    Create model - same as before but with debugging support
    '''
    print("Creating Model...")
    
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def create_data_loaders(data_dir, batch_size):
    '''
    Create data loaders - same as before
    '''
    print("Creating Data Loaders...")
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), 
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), 
        transform=test_transform
    )

    # (Optional) visibility—these end up in CloudWatch
    print(f"num_classes(train)={len(train_map)}")
    if len(test_dataset.targets):
        print(f"max_test_target={max(test_dataset.targets)}")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

def main(args):
    '''
    Main function with debugging and profiling hooks
    '''
    print("Starting Training with Debugging and Profiling...")
    
    # Create debugging hook
    # This is like attaching sensors to monitor your model
    hook = smd.Hook.create_from_json_file()
    
    # Get number of classes
    sample_loader, _ = create_data_loaders(args.data_dir, 32)
    num_classes = len(sample_loader.dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    model = net(num_classes)
    
    # Register the model with the hook
    # This tells the hook to monitor this model
    hook.register_hook(model)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
    
    # Create loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    # Register optimizer with hook
    hook.register_loss(loss_criterion)
    
    # Train the model
    model = train(model, train_loader, loss_criterion, optimizer, args.epochs, hook)
    
    # Test the model
    test_accuracy = test(model, test_loader, loss_criterion, hook)
    
    # Save the model
    model_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Also save the complete model for inference
    inference_model_path = os.path.join(args.model_dir, 'model_complete.pth')
    torch.save(model, inference_model_path)
    print(f"Complete model saved to {inference_model_path}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters (use best ones from tuning)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    
    # SageMaker arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_DATA'))
    
    args = parser.parse_args()
    
    main(args)