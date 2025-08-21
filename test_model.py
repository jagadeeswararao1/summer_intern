#!/usr/bin/env python3
"""
Test script to verify the model loading and prediction functionality
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Define the CNN model architecture (same as in app.py)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 2, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.dropout2d = nn.Dropout2d()
        self.fc = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("Testing model loading...")
    
    # Check if model file exists
    if not os.path.exists('model.pt'):
        print("❌ Error: model.pt file not found!")
        return False
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load model
        model = CNN().to(device)
        model.load_state_dict(torch.load('model.pt', map_location=device))
        model.eval()
        
        print("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_prediction():
    """Test if the model can make predictions"""
    print("\nTesting prediction functionality...")
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        model = CNN().to(device)
        model.load_state_dict(torch.load('model.pt', map_location=device))
        model.eval()
        
        # Define transform
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        # Create a dummy image for testing
        dummy_image = Image.new('RGB', (100, 100), color='red')
        image_tensor = transform(dummy_image)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probability = output.item()
            prediction = 1 if probability > 0.5 else 0
        
        print(f"✅ Prediction successful!")
        print(f"   Probability: {probability:.4f}")
        print(f"   Prediction: {prediction}")
        return True
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("Histopathological Cancer Detection - Model Test")
    print("=" * 50)
    
    # Test model loading
    model_loaded = test_model_loading()
    
    # Test prediction if model loaded successfully
    if model_loaded:
        prediction_works = test_prediction()
        
        if prediction_works:
            print("\n🎉 All tests passed! The model is ready to use.")
            print("You can now run the Flask app with: python app.py")
        else:
            print("\n❌ Prediction test failed. Check the model architecture.")
    else:
        print("\n❌ Model loading failed. Check the model file and architecture.")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 