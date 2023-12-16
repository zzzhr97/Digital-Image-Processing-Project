# https://blog.csdn.net/hxxjxw/article/details/108027696
# 输入图片的size是96*96
import numpy as np
import torch
from torch import nn,optim
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import torchvision
from datetime import datetime

BLOCK_LAYERS_1 = [6, 12, 24, 16] # DenseNet-121
BLOCK_LAYERS_2 = [6, 12, 32, 32] # DenseNet-169
BLOCK_LAYERS_3 = [6, 12, 48, 32] # DenseNet-201
BLOCK_LAYERS_4 = [6, 12, 36, 24] # DenseNet-161

def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer


# 稠密块由多个conv_block 组成，每块使⽤用相同的输出通道数。但在前向计算时，我们将每块的输入和输出在通道维上连结。
class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(
                conv_block(in_channel=channel, out_channel=growth_rate)
            )
            channel += growth_rate
            self.net = nn.Sequential(*block)

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x

if __name__ == '__main__':  
    blk = dense_block(in_channel=3, growth_rate=10, num_layers=4)
    X = torch.rand(4, 3, 8, 8)
    Y = blk(X)
    print(Y.shape) # torch.Size([4, 43, 8, 8])
 
 
def transition_block(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.AvgPool2d(2, 2)
    )
    return trans_layer
 

if __name__ == '__main__':
    blk = transition_block(in_channel=43, out_channel=10)
    print(blk(Y).shape) # torch.Size([4, 10, 4, 4])
 
 
class DenseNet(nn.Module):
    def __init__(self, in_channel=3, num_classes=2, growth_rate=32, block_layers=BLOCK_LAYERS_1):
        super(DenseNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        channels = 64
        block = []
        for i, layers in enumerate(block_layers):
            block.append(dense_block(channels, growth_rate, layers))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                block.append(transition_block(channels, channels // 2)) # 通过 transition 层将大小减半， 通道数减半
                channels = channels // 2
        self.block2 = nn.Sequential(*block)
        self.block2.add_module('bn', nn.BatchNorm2d(channels))
        self.block2.add_module('relu', nn.ReLU(True))
        self.block2.add_module('avg_pool', nn.AvgPool2d(3))
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
    

def DenseNet121(in_channel=3, num_classes=2, pretrained=False, include_top=True):
    weights = torchvision.models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = torchvision.models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def DenseNet169(in_channel=3, num_classes=2, pretrained=False, include_top=True):
    weights = torchvision.models.DenseNet169_Weights.DEFAULT if pretrained else None
    model = torchvision.models.densenet169(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def DenseNet201(in_channel=3, num_classes=2, pretrained=False, include_top=True):
    weights = torchvision.models.DenseNet201_Weights.DEFAULT if pretrained else None
    model = torchvision.models.densenet201(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def DenseNet161(in_channel=3, num_classes=2, pretrained=False, include_top=True):
    weights = torchvision.models.DenseNet161_Weights.DEFAULT if pretrained else None
    model = torchvision.models.densenet161(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def get_acc(output, label):
    total = output.shape[0]
    # output是概率，每行概率最高的就是预测值
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

if __name__ == '__main__':
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=96),
        torchvision.transforms.ToTensor()
    ])
    
    train_set = torchvision.datasets.CIFAR10(
        root='dataset/',
        train=True,
        download=True,
        transform=transform
    )
    
    # hand-out留出法划分
    train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000])
    
    test_set = torchvision.datasets.CIFAR10(
        root='dataset/',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False
    )

    # in_channel表示输入数据的深度，对于彩色照片通道数为3（RGB），对于黑白照片通道数为1
    net = DenseNet(in_channel=3, num_classes=10)
    
    lr = 1e-2
    optimizer = optim.SGD(net.parameters(), lr=lr)
    critetion = nn.CrossEntropyLoss()
    net = net.to(device)
    prev_time = datetime.now()
    valid_data = val_loader
    
    for epoch in range(3):
        train_loss = 0
        train_acc = 0
        net.train()
    
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # forward
            outputs = net(inputs)
            loss = critetion(outputs, labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
            train_acc += get_acc(outputs, labels)
            # 最后还要求平均的
    
        # 显示时间
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        # time_str = 'Time %02d:%02d:%02d'%(h,m,s)
        time_str = 'Time %02d:%02d:%02d(from %02d/%02d/%02d %02d:%02d:%02d to %02d/%02d/%02d %02d:%02d:%02d)' % (
            h, m, s, prev_time.year, prev_time.month, prev_time.day, prev_time.hour, prev_time.minute, prev_time.second,
            cur_time.year, cur_time.month, cur_time.day, cur_time.hour, cur_time.minute, cur_time.second)
        prev_time = cur_time
    
        # validation
        with torch.no_grad():
            net.eval()
            valid_loss = 0
            valid_acc = 0
            for inputs, labels in valid_data:
                inputs = inputs.to(device)
                labels = labels.to(device)
    
                outputs = net(inputs)
                loss = critetion(outputs, labels)
                valid_loss += loss.item()
                valid_acc += get_acc(outputs, labels)
    
        print("Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f,"
            % (epoch, train_loss / len(train_loader), train_acc / len(train_loader), valid_loss / len(valid_data),
                valid_acc / len(valid_data))
            + time_str)
    
        torch.save(net.state_dict(), 'checkpoints/params.pkl')
    
    # 测试
    with torch.no_grad():
        net.eval()
        correct = 0
        total = 0
        for (images, labels) in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = net(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        print("The accuracy of total {} val images: {}%".format(total, 100 * correct / total))
