import os
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms

print("downloading train dataset....")
train_dataset = datasets.MNIST(
    root="data", train=True, download=False, transform=transforms.ToTensor()
)
print("downloading test dataset....")
test_dataset = datasets.MNIST(
    root="data", train=False, download=False, transform=transforms.ToTensor()
)

print("loading train dataset....")
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)

print("loading test dataset....")
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=64, shuffle=True
)


# Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


print("Creating model....")
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(2):
    print(f"Epoch {epoch}")
    for batch_idx, (data, targets) in enumerate(train_loader):

        data = data
        targets = targets

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}", end=" ")
            print(f"Loss: {loss.item()}")

        # gradient descent or adam step
        optimizer.step()

# Testing


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    print(f"Checking accuracy on {loader.dataset}")

    with torch.no_grad():
        for x, y in loader:
            x = x
            y = y
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
