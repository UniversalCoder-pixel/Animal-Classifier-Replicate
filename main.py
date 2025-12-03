import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# === PATHS ===
DATASET_PATH = "dataset"  # adjust if needed
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === LOAD DATA ===
train_data = datasets.ImageFolder(TRAIN_PATH, transform=transform)
test_data = datasets.ImageFolder(TEST_PATH, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

num_classes = len(train_data.classes)
print("Detected classes:", train_data.classes)

# === MODEL ===
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

# === TRAINING SETUP ===
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)

# === TRAIN LOOP ===
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

# === SAVE MODEL ===
torch.save(model.state_dict(), "animal_model.pth")
print("Model saved as animal_model.pth")

