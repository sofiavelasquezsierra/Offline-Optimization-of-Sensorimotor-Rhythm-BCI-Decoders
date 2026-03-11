import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, nb_classes=2, Chans=10, Samples=400, dropoutRate=0.5):
        super(EEGNet, self).__init__()
        
        # Layer 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(8)
        
        # Layer 2: Depthwise Conv (Spatial Filter)
        self.depthwise = nn.Conv2d(8, 16, (Chans, 1), groups=8, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.act1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)
        
        # Layer 3: Separable Conv
        self.separable = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.act2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)
        
        # Layer 4: Classification
        self.flatten = nn.Flatten()
        
        self.fc = nn.Linear(192, nb_classes) 

    def forward(self, x):
        # Input shape: (Batch, 1, Channels, Time)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.act1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        x = self.separable(x)
        x = self.batchnorm3(x)
        x = self.act2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        return x