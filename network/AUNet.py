import torch
from torch import nn

class DownsampleLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Params:
            x: input
        Returns:
            out: input for next downsample layer
            out_cp: input for upsample layer
        """
        out_cp = self.Conv_BN_ReLU_2(x)
        out = self.downsample(out_cp)
        return out, out_cp

class UpSampleLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel * 2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channel * 2, out_channels=out_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x, x_prev):
        '''
        Params:
            x: input
            out: input from downsample layer
        Returns:
            out: input for upsample layer
            out_2: input for next downsample layer
        '''
        out=self.Conv_BN_ReLU_2(x)
        out=self.upsample(out)
        out=torch.cat((out,x_prev),dim=1)
        return out

class AUNet(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, config=None):
        super(AUNet, self).__init__()

        # downsample
        self.down_layers = nn.ModuleDict() 
        self.down_layers['down_0'] = DownsampleLayer(in_dim, config[0])
        for i in range(1, len(config)-1):
            self.down_layers[f'down_{i}'] = DownsampleLayer(config[i-1], config[i])

        # down + up
        self.down_up_layers = UpSampleLayer(config[-2], config[-2])

        # upsample
        self.up_layers = nn.ModuleDict()
        for i in range(0, len(config)-2):
            self.up_layers[f'up_{i}'] = UpSampleLayer(config[-i-1], config[-i-3])

        self.up_layer_last = nn.Sequential(
            nn.Conv2d(config[1], config[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(config[0]),
            nn.ReLU(),
            nn.Conv2d(config[0], config[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(config[0]),
            nn.ReLU(),
        )

        # output layer
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(config[0], out_dim),
        ) 

    def forward(self,x):
        # downsample
        x_cp = {}
        for i  in range(0, len(self.down_layers)):
            x, x_cp[f'down_{i}'] = self.down_layers[f'down_{i}'](x)

        # down + up
        x = self.down_up_layers(x, x_cp[f'down_{len(self.down_layers)-1}'])

        # upsample
        for i in range(0, len(self.up_layers)):
            x = self.up_layers[f'up_{i}'](x, x_cp[f'down_{len(self.down_layers)-2-i}'])
        x = self.up_layer_last(x)

        # output
        x = self.output_layer(x)
        return x
    
def get_AUNet(in_dim, out_dim, first_dim, depth):
    config = [first_dim * 2**i for i in range(depth)]
    return AUNet(in_dim, out_dim, config)

def AUNet3(num_classes=2, in_channel=3, pretrained=False):
    # 512 -> 256 -> 128 -> 256 -> 512
    return get_AUNet(in_channel, num_classes, 16, 3)

def AUNet4(num_classes=2, in_channel=3, pretrained=False):
    # 512 -> 256 -> 128 -> 64 -> 128 -> 256 -> 512
    return get_AUNet(in_channel, num_classes, 32, 4)

def AUNet5(num_classes=2, in_channel=3, pretrained=False):
    # 512 -> 256 -> 128 -> 64 -> 32 -> 64 -> 128 -> 256 -> 512
    return get_AUNet(in_channel, num_classes, 32, 5)

def AUNet6(num_classes=2, in_channel=3, pretrained=False):
    # 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512
    return get_AUNet(in_channel, num_classes, 32, 6)
