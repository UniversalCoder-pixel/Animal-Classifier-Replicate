# predict.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown
import os

# Google Drive link for your model
MODEL_URL = "https://drive.google.com/uc?id=1BeBfZ-Dhi7bzSFBIlZ-jAjHGb-sW_ZK8"
MODEL_PATH = "animal_model.pth"

# Download the model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Define model architecture exactly like during training
model = models.resnet18(pretrained=False)
num_classes = 7  # adjust to your number of animal classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load state dictionary
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Set model to evaluation mode
model.eval()

# Define the classes
classes = ['Cat', 'Cow', 'Deer', 'Dog', 'Rabbit', 'Sheep', 'elephant']

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_name = classes[predicted.item()]
    return class_name

