import torch.nn as nn

class SResNet_block(nn.Module):
    def __init__(self, input_depth, output_depth):
        super(SResNet_block, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_depth, output_depth, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_depth),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(output_depth, output_depth, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_depth),
            nn.ReLU()
        )

    def forward(self, x):
        x_0 = x
        x = self.block1(x)
        x = self.block2(x)

        if x_0.shape[1] == x.shape[1]:
            x = x + x_0     # 不能使用 x += x_0!

        return x
    
class SResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SResNet, self).__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                SResNet_block(3, 8),    # (3, 512, 512) -> (8, 512, 512)
                SResNet_block(8, 8),    # (8, 512, 512) -> (8, 512, 512)
                nn.MaxPool2d(2, 2)      # (8, 512, 512) -> (8, 256, 256)
            ),
            nn.Sequential(
                SResNet_block(8, 16),   # (8, 256, 256) -> (16, 256, 256)
                SResNet_block(16, 16),  # (16, 256, 256) -> (16, 256, 256)
                nn.MaxPool2d(2, 2)      # (16, 256, 256) -> (16, 128, 128)
            ),
            nn.Sequential(
                SResNet_block(16, 32),  # (16, 128, 128) -> (32, 128, 128)
                SResNet_block(32, 32),  # (32, 128, 128) -> (32, 128, 128)
                nn.MaxPool2d(2, 2)      # (32, 128, 128) -> (32, 64, 64)
            ),
            nn.Sequential(
                SResNet_block(32, 32),  # (32, 64, 64) -> (32, 64, 64)
                SResNet_block(32, 32),  # (32, 64, 64) -> (32, 64, 64)
                nn.MaxPool2d(2, 2)      # (32, 64, 64) -> (32, 32, 32)
            ),
        ])

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*32, 160),
            nn.ReLU(),
            nn.Linear(160, num_classes)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.fc(x)
        return x