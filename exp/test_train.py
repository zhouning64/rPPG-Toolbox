import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Define the dataset class
class RandomVideoDataset(Dataset):
    def __init__(self, num_samples=1000, num_frames=16, height=128, width=128, num_classes=10):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random video data
        video = torch.rand(3, self.num_frames, self.height, self.width)  # [C, T, H, W]
        label = torch.randint(0, self.num_classes, (1,)).item()  # Random class label
        return video, label


# Define the model
class VideoActionModel(nn.Module):
    def __init__(self, num_classes=10):
        super(VideoActionModel, self).__init__()
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=False),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=False),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
        )
        self.fc = nn.Linear(64, num_classes)  # Fully connected layer for classification

    def forward(self, x):
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = x.mean([2, 3, 4])  # Global average pooling over time and space
        x = self.fc(x)
        return x


# Initialize dataset, dataloader, model, optimizer, and loss function
num_classes = 10
batch_size = 8
learning_rate = 0.001
num_epochs = 20

dataset = RandomVideoDataset(num_samples=100, num_classes=num_classes)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = VideoActionModel(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model.to(device)

criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for videos, labels in dataloader:
        videos, labels = videos.to(device), labels.to(device)

        # Forward pass
        outputs = model(videos)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Print epoch results
    print(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"Loss: {epoch_loss/len(dataloader):.4f}, "
        f"Accuracy: {correct/total*100:.2f}%"
    )

print("Training complete.")