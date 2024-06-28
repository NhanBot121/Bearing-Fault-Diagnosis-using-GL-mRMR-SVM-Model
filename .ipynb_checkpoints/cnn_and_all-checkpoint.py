import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 50
batch_size = 20
learning_rate = 0.0015
drop_out = 0.3

# CNN model
class ConvNet(nn.Module):
    def __init__(self, drop_out=0.3):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*2*2, 128)
        self.dropout = nn.Dropout(drop_out)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*2*2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        #return F.softmax(x, dim=1)  # Apply softmax to get probabilities
        return x

def train_test_loader(df, random_state):
    features = df.columns[1:]
    label = df.columns[0]
    X_train, X_test, y_train, y_test = train_test_split(df[features],
                                                        df[label],
                                                        train_size=1000, stratify=df['label'], random_state=random_state,
                                                        shuffle=True)

    X_train = X_train.to_numpy().reshape(-1, 1, 32, 32)
    X_test = X_test.to_numpy().reshape(-1, 1, 32, 32)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Load and preprocess data
def data_loader(df, batch_size=1180):
    features = df.columns[1:]
    labels = df.columns[0]

    X = df[features].to_numpy().reshape(-1, 1, 32, 32)
    y = df[labels].to_numpy()

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader



# Training function
def train_cnn(model, train_loader, criterion, optimizer):
    model.train()
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 2000 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Evaluation function
def evaluate_cnn(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classication Report
    class_report = classification_report(all_labels, all_preds, output_dict=True)
    print('Classification Report:')
    print(class_report)

    return cm, class_report


