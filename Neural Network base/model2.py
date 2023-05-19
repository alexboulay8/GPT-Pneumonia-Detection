import requests
import os
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor

# Define the URL of the MedMNIST dataset in NPZ format
url = 'https://zenodo.org/record/6496656/files/pneumoniamnist.npz?download=1'

# Define the path where the dataset will be downloaded and extracted
download_path = r'C:\Users\mgbou\OneDrive\Documents\GitHub\GPT-Pneumonia-Detection'

# Create the download directory if it doesn't exist
os.makedirs(download_path, exist_ok=True)

# Download the dataset file
response = requests.get(url)
file_path = os.path.join(download_path, 'pneumoniamnist.npz')
with open(file_path, 'wb') as file:
    file.write(response.content)

# Extract the dataset
extract_path = os.path.join(download_path, 'pneumoniamnist')
os.makedirs(extract_path, exist_ok=True)

# Load the dataset from the NPZ file
data = np.load(file_path)

# Save each data split as separate files
for split in ['train', 'val', 'test']:
    np.savez(os.path.join(extract_path, f'{split}.npz'), x=data[f'/{split}/images'], y=data[f'/{split}/labels'])

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MedMNIST
from torch.utils.data.sampler import SubsetRandomSampler

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define the layers of your model
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 12 * 12, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the transformation applied to the images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MedMNIST dataset
dataset = MedMNIST(root=r'C:\Users\mgbou\OneDrive\Documents\GitHub\GPT-Pneumonia-Detection\pneumoniamnist.npz', split='train', transform=transform, download=True)

# Split the dataset into training and validation sets
val_split = 0.2
num_train = len(dataset)
indices = list(range(num_train))
split = int(np.floor(val_split * num_train))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Create data loaders for training and validation
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

# Instantiate the model and move it to the device
model = Net().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute training accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        train_loss += loss.item()

    train_accuracy = 100.0 * correct / total
    print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.2f}%')

# Prediction module
def predict_image(model, image_path):
    model.eval()
    image = Image.open(image_path).convert('L')  # Load the image as grayscale
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = outputs.max(1)
    return predicted.item()

# Usage example
test_image_path = 'path/to/test/image.jpg'
prediction = predict_image(model, test_image_path)
print(f'Prediction: {"Pneumonia" if prediction == 1 else "Non-Pneumonia"}')
