import torch.nn as nn
import torch

class VGGNet(nn.Module):
    def __init__(self, features, num_classes=2, init_weights=False):
        super(VGGNet, self).__init__()
        self.features = features
        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )
        # initialize
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)                # N*3*224*224
        x = torch.flatten(x, start_dim=1)   # N*512*7*7
        x = self.classifier(x)              # N*512*7*7
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0) 


# extract features
def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:  
        if v == "M":    # Max pooling 
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:           # Conv
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v  
    return nn.Sequential(*layers)

# feature extraction part of vgg
cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def VGG(model_name='VGG16', **kwargs):
    try:
        cfg = cfgs[model_name] # 得到vgg16对应的列表
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    # 搭建模型
    model = VGGNet(make_features(cfg), **kwargs)
    return model

def VGG11(num_classes=2):
    return VGG('VGG11', num_classes=num_classes)

def VGG13(num_classes=2):
    return VGG('VGG13', num_classes=num_classes)

def VGG16(num_classes=2):
    return VGG('VGG16', num_classes=num_classes)

def VGG19(num_classes=2):
    return VGG('VGG19', num_classes=num_classes)