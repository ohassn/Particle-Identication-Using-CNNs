import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Deep Learning Model
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_class=2, D=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16*25*25, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


# Setting Device
device = torch.device('cuda')

#  Hyper-parameters
in_channel = 1  # number of input channels
num_classes = 2  # number of particle types
learning_rate = 0.001  # tweak
batch_size = 64  # number of samples before the parameters are updated
num_epochs = 100  # number of full passes

#  Loading Datasets

full_dataset_pi = torch.load("Datasets (no mag)/DENSE_pi+_5GeV_3deg_50deg_1e5.edm4hep.pt")[2]
full_dataset_e = torch.load("Datasets (no mag)/DENSE_e-_5GeV_3deg_50deg_1e5.edm4hep.pt")[2]

full_dataset = full_dataset_pi + full_dataset_e 
print(len(full_dataset_pi), len(full_dataset_e), len(full_dataset))
random.shuffle(full_dataset)

train_dataset = full_dataset[:int(len(full_dataset)*0.8)]  # path for file
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = full_dataset[int(len(full_dataset)*0.8):]  # path for file
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize Network
model = CNN().to(device) 

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

loss_vals = []

# Train Network
for epoch in range(num_epochs):
    epoch_loss = []
    print(f"Epoch: {epoch}")
    total_loss = 0
    total_correct = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        # to cuda
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward        
        scores = model(data)
        loss = criterion(scores, targets)
        total_loss += loss.item()
        total_correct += get_num_correct(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()

        epoch_loss.append(loss.item())
    loss_vals.append(sum(epoch_loss)/len(epoch_loss))
    print(f'Epoch: {epoch}, Loss: {sum(epoch_loss)/len(epoch_loss)}')

model.eval()

# Test Accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            # scores.max(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    print(f'Accuracy = {num_correct} / {num_samples} =  {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
