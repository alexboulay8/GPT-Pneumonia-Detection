import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

dataset_name = 'pneumoniamnist'
download_data = True
learning_rate = 0.001
num_epochs = 3
batch_size = 32

dataset_info = INFO[dataset_name]
task_type = dataset_info['task']
num_channels = dataset_info['n_channels']
num_classes = len(dataset_info['label'])

DatasetClass = getattr(medmnist, dataset_info['python_class'])

data_preprocessing = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

training_dataset = DatasetClass(split='train', transform=data_preprocessing, download=download_data)
testing_dataset = DatasetClass(split='test', transform=data_preprocessing, download=download_data)

training_data_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
eval_data_loader = DataLoader(dataset=training_dataset, batch_size=2*batch_size, shuffle=False)
testing_data_loader = DataLoader(dataset=testing_dataset, batch_size=2*batch_size, shuffle=False)

training_dataset.montage(length=1)
training_dataset.montage(length=20)

class Network(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Network, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 128),  
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

network = Network(in_channels=num_channels, num_classes=num_classes)

criterion = nn.BCEWithLogitsLoss() if task_type == "multi-label, binary-class" else nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
training_loss = []
training_accuracy = []

for epoch in range(num_epochs):
    total_loss = 0.0
    total_correct = 0
    counter = 0
    network.train()
    
    for inputs, targets in tqdm(training_data_loader):
        optimizer.zero_grad()
        predictions = network(inputs)
        targets = targets.to(torch.float32) if task_type == 'multi-label, binary-class' else targets.squeeze().long()
        loss = criterion(predictions, targets)
        total_loss += loss.item()
        _, predicted = torch.max(predictions.data, 1)
        total_correct += (predicted == targets).sum().item()
        loss.backward()
        optimizer.step()
        counter += 1
        
    epoch_loss = total_loss / counter
    epoch_accuracy = 100. * (total_correct / len(training_data_loader.dataset))
    training_loss.append(epoch_loss)
    training_accuracy.append(epoch_accuracy)

def evaluate(split):
    network.eval()
    actual_labels = torch.tensor([])
    predicted_scores = torch.tensor([])
    data_loader = eval_data_loader if split == 'train' else testing_data_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            predictions = network(inputs)
            if task_type == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                predictions = predictions.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                predictions = predictions.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)
            actual_labels = torch.cat((actual_labels, targets), 0)
            predicted_scores = torch.cat((predicted_scores, predictions), 0)
        actual_labels = actual_labels.numpy()
        predicted_scores = predicted_scores.detach().numpy()
        
        evaluator = Evaluator(dataset_name, split)
        metrics = evaluator.evaluate(predicted_scores)
        print(f'{split}  auc: {metrics[0]:.3f}  acc: {metrics[1]:.3f}')

print('==> Evaluating ...')
evaluate('train')
evaluate('test')

torch.save(network, r"Path To Directory (.pth file)")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), training_loss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), training_accuracy, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
