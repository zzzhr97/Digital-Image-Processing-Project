import torch
from torch import nn
import torchvision.models as models

class TestNet(nn.Module):
    def __init__(self, num_classes, weights=None):
        super(TestNet, self).__init__()
        self.resnet = models.resnet34(weights=weights)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_features, out_features=1)

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),    # (3, 512, 512) -> (4, 512, 512)
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2),                            # (4, 512, 512) -> (4, 256, 256)      
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),    # (4, 256, 256) -> (8, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                            # (8, 256, 256) -> (8, 128, 128)      
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),    # (8, 128, 128) -> (8, 128, 128)     
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                            # (8, 128, 128) -> (8, 64, 64)
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),    # (8, 64, 64) -> (8, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                            # (8, 64, 64) -> (8, 32, 32)
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(8*32*32, 320),                             
            nn.ReLU(),          
            nn.Linear(320, num_classes)                                
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.shape[0], -1)
        x = self.linear_layer(x)
        return x