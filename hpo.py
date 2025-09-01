 #TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
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

# CRITICAL FIX: Enable loading of truncated/corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up logging to see what's happening
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class RobustImageFolder(torchvision.datasets.ImageFolder):
    """
    Custom ImageFolder that handles corrupted images by skipping them
    """
    def __getitem__(self, index):
        path, target = self.samples[index]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Try to open the image
                with Image.open(path) as sample:
                    # Convert to RGB to handle grayscale/RGBA
                    sample = sample.convert('RGB')
                    
                    # Verify the image is not corrupted by trying to load it
                    sample.load()
                    
                    if self.transform is not None:
                        sample = self.transform(sample)
                    if self.target_transform is not None:
                        target = self.target_transform(target)
                    
                    return sample, target
                    
            except Exception as e:
                print(f"Warning: Corrupted image {path}, attempt {attempt + 1}: {str(e)}")
                
                if attempt == max_retries - 1:
                    # Find a replacement image from the same class
                    same_class_indices = [i for i, (_, t) in enumerate(self.samples) if t == target and i != index]
                    if same_class_indices:
                        # Try a different image from the same class
                        replacement_index = same_class_indices[0]
                        path, target = self.samples[replacement_index]
                        print(f"Using replacement image: {path}")
                    else:
                        # Create a black placeholder image as last resort
                        print(f"Creating placeholder for corrupted image: {path}")
                        sample = Image.new('RGB', (224, 224), color='black')
                        if self.transform is not None:
                            sample = self.transform(sample)
                        return sample, target

def test(model, test_loader, criterion, device='cpu'):
    '''
    This function tests our model and returns accuracy and loss
    Think of this as giving your model a final exam
    '''
    print("Testing Model on Whole Testing Dataset")
    
    model.eval()  # Put model in evaluation mode (no learning)
    running_loss = 0
    running_corrects = 0
    
    # Turn off gradient calculation for testing (saves memory)
    with torch.no_grad():
        for inputs, labels in test_loader:
            try:
                # Move data to device (GPU/CPU)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass - just get predictions
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Get the predicted class (highest probability)
                _, preds = torch.max(outputs, 1)
                
                # Keep track of loss and correct predictions
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
            except Exception as e:
                print(f"Error in test batch: {str(e)}")
                continue
    
    # Calculate averages
    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    
    print(f"Testing Loss: {total_loss}")
    print(f"Testing Accuracy: {total_acc}")
    
    return total_acc

def train(model, train_loader, criterion, optimizer, epochs, device):
    '''
    This function trains our model
    Think of this as teaching your model by showing it lots of examples
    '''
    print("Training Model...")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        model.train()  # Put model in training mode
        running_loss = 0.0
        running_corrects = 0
        successful_batches = 0
        
        # Go through all batches of training data
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            try:
                # Move data to device (GPU/CPU)
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Clear old gradients
                optimizer.zero_grad()
                
                # Forward pass - get model predictions
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass - calculate gradients
                loss.backward()
                
                # Update model parameters
                optimizer.step()
                
                # Track statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                successful_batches += 1
                
                # Print progress every 50 batches
                if batch_idx % 50 == 0:
                    print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)} - Skipping batch")
                continue
        
        # Calculate epoch statistics
        if successful_batches > 0:
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects / len(train_loader.dataset)
            print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        else:
            print(f"Epoch {epoch+1} - No successful batches!")
    
    return model
    
def net(num_classes):
    '''
    This function creates our model using a pre-trained ResNet50
    Think of this as taking a smart person who already knows a lot,
    and teaching them your specific task
    '''
    print("Creating Model...")
    
    # Load pre-trained ResNet50 model
    # Note: weights parameter is preferred over pretrained in newer PyTorch versions
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    except:
        # Fallback for older PyTorch versions
        model = models.resnet50(pretrained=True)
    
    # Freeze early layers (they already know basic features)
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final layer to match our number of classes
    # This is like changing the final decision-making part
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Make sure the final layer is trainable
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model

def create_data_loaders(data_dir, batch_size):
    '''
    This function creates data loaders that feed images to our model
    Think of this as organizing your study materials into manageable chunks
    '''
    print("Creating Data Loaders...")
    print(f"Data directory: {data_dir}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    
    # Check if train and test directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory {train_dir} does not exist")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory {test_dir} does not exist")
    
    # Define transformations for our images
    # These help make our model more robust
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),      # Add explicit resize first
        transforms.RandomResizedCrop(224),  # Randomly crop and resize
        transforms.RandomHorizontalFlip(),  # Sometimes flip horizontally
        transforms.ToTensor(),              # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # Normalize like ImageNet
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),      # Add explicit resize first
        transforms.CenterCrop(224),         # Crop center
        transforms.ToTensor(),              # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # Normalize like ImageNet
    ])
    
    # Create datasets using our robust ImageFolder
    train_dataset = RobustImageFolder(
        train_dir, 
        transform=train_transform
    )
    
    test_dataset = RobustImageFolder(
        test_dir, 
        transform=test_transform
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 for SageMaker compatibility
        drop_last=True  # Drop incomplete batches to avoid size issues
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,  # Set to 0 for SageMaker compatibility
        drop_last=False
    )
    
    return train_loader, test_loader

def main(args):
    '''
    Main function that orchestrates everything
    '''
    print("Starting Training Process...")
    print(f"Arguments: {args}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Validate arguments
    if not args.data_dir:
        raise ValueError("Data directory not specified")
    if not args.model_dir:
        raise ValueError("Model directory not specified")
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    try:
        # Get number of classes from the dataset
        sample_loader, _ = create_data_loaders(args.data_dir, 32)
        num_classes = len(sample_loader.dataset.classes)
        print(f"Number of classes: {num_classes}")
        
        # Initialize model and move to device
        model = net(num_classes)
        model = model.to(device)
        
        # Create data loaders with specified batch size
        train_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)
        
        # Create loss function and optimizer
        loss_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
        
        print(f"Model parameters to train: {sum(p.numel() for p in model.fc.parameters())}")
        
        # Train the model
        model = train(model, train_loader, loss_criterion, optimizer, args.epochs, device)
        
        # Test the model
        test_accuracy = test(model, test_loader, loss_criterion, device)
        
        # Save the trained model
        model_path = os.path.join(args.model_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Also save model info
        model_info = {
            'num_classes': num_classes,
            'class_names': sample_loader.dataset.classes,
            'test_accuracy': test_accuracy
        }
        
        info_path = os.path.join(args.model_dir, 'model_info.pth')
        torch.save(model_info, info_path)
        print(f"Model info saved to {info_path}")
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters we want to tune
    parser.add_argument('--lr', type=float, default=0.01, 
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=2, 
                       help='Number of epochs (default: 2)')
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_DATA'))
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"Script failed with error: {str(e)}")
        sys.exit(1)