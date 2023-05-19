import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()
        # Define your model architecture here
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(1024, 128),  # Adjusted input size to 1024
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def predict_image(model, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert('L')
    image_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    image = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
    _, predicted = outputs.max(1)
    return predicted.item()


# Load the saved model/load directory path to where the .pth file is located on your computer
model = torch.load(r'C:\Users\mgbou\OneDrive\Documents\GitHub\GPT-Pneumonia-Detection\pneumonia_model.pth')
model.eval()

# Directory containing the test images/Change to directory on your computer
test_images_dir = r"C:\Users\mgbou\OneDrive\Documents\GitHub\GPT-Pneumonia-Detection\test_images\Non-Pneumonia"

# Process each image file in the directory
for filename in os.listdir(test_images_dir):
    if filename.endswith(".jpeg") or filename.endswith(".jpg"):
        image_path = os.path.join(test_images_dir, filename)
        prediction = predict_image(model, image_path)
        print(f"Image: {filename} - Prediction: {'Pneumonia' if prediction == 1 else 'Non-Pneumonia'}")
