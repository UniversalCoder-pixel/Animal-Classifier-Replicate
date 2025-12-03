# predict.py
import torch
from torchvision import transforms, models
from PIL import Image

CLASSES = ['Cat', 'Cow', 'Deer', 'Dog', 'Rabbit', 'Sheep', 'elephant']
MODEL_PATH = "animal_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
    return CLASSES[pred.item()]

if __name__ == "__main__":
    test_image = "test.jpg"  # replace with an actual file
    print("Prediction:", predict_image(test_image))
