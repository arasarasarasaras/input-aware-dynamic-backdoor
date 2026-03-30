import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, final=False):
        super().__init__()
        
        if final:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=inChannels, out_channels=outChannels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(outChannels, eps=1e-5, momentum=0.05, affine=True),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)
    
class DownSampleBlock(nn.Module):
    def __init__(self, kernel_size=2 , stride=2):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.maxpool(x)
    
class UpSampleBlock(nn.Module):
    def __init__(self, scale_factor=2 , mode="bilinear", align_corners=False):
        super().__init__()
        self.upscale = nn.Upsample(
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners if mode in ["bilinear", "bicubic", "trilinear"] else None
        )

    def forward(self, x):
        return self.upscale(x)
    
class PreActBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.batch_norm1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.batch_norm2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(
                in_planes,
                planes * self.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.batch_norm1(x))
        shortcut = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else x

        out = self.conv1(out)
        out = F.relu(self.batch_norm2(out))
        out = self.conv2(out)
        out += shortcut
        return out