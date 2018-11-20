# coding: utf-8
import pandas as pd
import numpy as np
import torch
import torch.cuda
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def plot_confusion_matrix(y_pred, y_true):
    mtx = confusion_matrix(y_pred, y_true)
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  cbar=False, ax=ax)
    #  square=True,
    plt.xlabel('true label')
    plt.ylabel('predicted label')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_csv = pd.read_csv("train.csv")
test_csv = pd.read_csv("test.csv")

y = train_csv['label']
X = train_csv.drop(["label"],axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


num_epochs = 20
batch_size = 100
learning_rate = 0.001

class MNISTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X.values.reshape(-1, 28, 28, 1)).permute(0, 3, 1, 2)
        self.y = list(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


train_dataset = MNISTDataset(X_train, y_train)
test_dataset = MNISTDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Linear(7 * 7 * 32, 10)
    def forward(self, X):
        out = self.convnet(X)
        out = out.reshape(out.size(0), -1)
        return self.fc(out)

model = MNISTConvNet().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    y_pred = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        y_pred.extend(np.asarray(predicted.cpu()))
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)

submit_dataset = MNISTDataset(test_csv, np.zeros(len(test_csv)))
submit_loader = DataLoader(test_dataset, shuffle=False, batch_size=100)

predictions = []
model.eval()
with torch.no_grad():
    for images, labels in submit_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        predictions.extend(np.asarray(predicted.cpu()))

submission = pd.DataFrame(data={'ImageId': range(1, len(predicitions) + 1), 'Label': predicitions})
submission.to_csv("submission.csv", index=False)