import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import PreActBlock

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.batch_norm = nn.BatchNorm2d(512 * block.expansion)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []

        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes* block.expansion

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)         # 32x32
        out = self.layer1(out)      # 32x32
        out = self.layer2(out)      # 16x16
        out = self.layer3(out)      # 8x8
        out = self.layer4(out)      # 4x4
        out = F.relu(self.batch_norm(out))
        out = F.avg_pool2d(out, 4)  # 1x1
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    
def PreActResNet18(num_classes=10, in_channels=3):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes, in_channels=in_channels)