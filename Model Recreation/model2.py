import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from medmnist.dataset import PneumoniaMNIST

# Transform to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # normalize inputs to [-1, 1]
])

# Load the datasets
train_dataset = PneumoniaMNIST(root='.', split='train', transform=transform, download=True)
val_dataset = PneumoniaMNIST(root='.', split='val', transform=transform, download=True)
test_dataset = PneumoniaMNIST(root='.', split='test', transform=transform, download=True)

# Set up data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

class PneumoniaNet(nn.Module):
    def __init__(self):
        super(PneumoniaNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)  # 64 channels, and 7x7 image size
        self.fc2 = nn.Linear(128, 1)  # binary classification: pneumonia or no pneumonia

    def forward(self, x):
        # Layer 1: Convolutional + ReLU + Max Pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Layer 2: Convolutional + ReLU + Max Pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten
        x = x.view(x.size(0), -1)

        # Layer 3: Fully connected + ReLU
        x = F.relu(self.fc1(x))

        # Layer 4: Fully connected
        x = self.fc2(x)

        return x




import torch.optim as optim

# Assuming you have defined your model as an instance of the previously defined PneumoniaNet
model = PneumoniaNet()

# Defining the loss function
# Binary Cross Entropy with Logits includes a sigmoid activation on the model outputs
# as well as binary cross entropy loss after that
criterion = nn.BCEWithLogitsLoss()

# Defining the optimizer (Stochastic Gradient Descent)
# You may want to adjust the learning rate based on your specific problem and model
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Number of epochs
num_epochs = 10

# Loop over the dataset multiple times
for epoch in range(num_epochs):
    running_loss = 0.0
    val_loss = 0.0

    # Training loop
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)
        # compute loss
        loss = criterion(outputs, labels.view(-1, 1).type_as(outputs))
        # backward pass
        loss.backward()
        # optimize (update weights)
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
    # Normalizing the loss by the total number of train batches
    running_loss /= len(train_loader)
    print(f"Training Loss: {running_loss}")
    
    # Validation loop
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1).type_as(outputs))
            val_loss += loss.item()
    
    # Normalizing the loss by the total number of val batches
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss}")

print('Finished Training')

# Assuming the model is called 'model'
torch.save(model.state_dict(), './pneumonia_net.pth')

# Now let's define a function to make predictions with our trained model
def predict_image(image, model):
    # Change the model to evaluation mode. This is necessary as certain operations like dropout
    # behave differently during training and testing.
    model.eval()

    # Convert to a batch of 1
    image = image.unsqueeze(0)

    # Get predictions from model
    output = model(image)
    
    # Convert output probabilities to predicted class (0 or 1)
    _, preds = torch.max(output, 1)
    
    return preds.item()

# Now let's load the model and use it for inference
def load_model_and_predict(test_images):
    # First, load the model
    model = PneumoniaNet()  # We are assuming the model's class is PneumoniaNet
    model.load_state_dict(torch.load('./pneumonia_net.pth'))

    # Now let's use our model to predict the class of the test images
    for i, (image, label) in enumerate(test_images):
        # Image is already a tensor, no need to convert it
        prediction = predict_image(image, model)
        print(f"Test Image {i}: Predicted Class: {prediction}, True Class: {label}")

# Assuming `test_loader` is your test dataset loader
load_model_and_predict(test_loader)


