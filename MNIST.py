
# coding: utf-8

# In[1]:


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


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

train_csv = pd.read_csv("train.csv")
test_csv = pd.read_csv("test.csv")

y = train_csv['label']
X = train_csv.drop(["label"],axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class MNISTDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.values.astype('float32') / 255
        means = np.mean(self.X, axis=1)
        constant = 10
        variances = np.var(self.X, axis=1) + constant
        self.X = (self.X.T - means).T
        self.X = (self.X.T / np.sqrt(variances)).T
        self.X = torch.Tensor(self.X.reshape(-1, 28, 28, 1)).permute(0, 3, 1, 2)
        self.y = list(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


train_dataset = MNISTDataset(X_train, y_train)
test_dataset = MNISTDataset(X_test, y_test)

batch_size = 128
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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.resblock = nn.Sequential(
            conv1x1(in_planes, out_planes),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            
            conv3x3(out_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            
            conv1x1(out_planes, out_planes * self.expansion),
            nn.BatchNorm2d(out_planes * self.expansion)
        )
        self.stride = stride
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.resblock(x)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)

class ResidualBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.resblock = nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes, out_planes),
            nn.ReLU(),
            
            conv3x3(out_planes, out_planes),
            nn.BatchNorm2d(out_planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.resblock(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, building_block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._build_residual_blocks(building_block, 64, layers[0])
        self.layer2 = self._build_residual_blocks(building_block, 128, layers[1], stride=2)
        self.layer3 = self._build_residual_blocks(building_block, 256, layers[2], stride=2)
        self.layer4 = self._build_residual_blocks(building_block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * building_block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _build_residual_blocks(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# model = MNISTConvNet().to(device)
model = ResNet(BottleneckBlock, [3, 4, 6, 3]).to(device)


learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=3e-5)


model.train()
num_epochs = 50
loss_history = []
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
        loss_history.append(loss.item())
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


plt.plot(loss_history)

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
submit_loader = DataLoader(submit_dataset, shuffle=False, batch_size=100)


predictions = []
model.eval()
with torch.no_grad():
    for images, labels in submit_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        predictions.extend(np.asarray(predicted.cpu()))



submission = pd.DataFrame(data={'ImageId': range(1, len(predictions) + 1), 'Label': predictions})
submission.to_csv("submission.csv", index=False)
