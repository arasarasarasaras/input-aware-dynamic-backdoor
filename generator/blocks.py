import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels, final=False):
        super().__init__()
        
        if final:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=inChannels, out_channels=outChannels, padding=1),
                nn.Sigmoid()
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=inChannels, out_channels=outChannels, stride=1, padding=1),
                nn.BatchNorm2d(outChannels, eps=1e-5, momentum=0.05, affine=True),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.block(x)
    
class DownSampleBlock(nn.Module):
    def __init__(self, kernel_size=2 , stride=2):
        super().__init__(kernel_size, stride)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.maxpool(x)
    
class UpSampleBlock(nn.Module):
    def __init__(self, scale_factor=2 , mode="bilinear", align_corners=False):
        super().__init__(scale_factor, mode, align_corners)
        self.upscale = nn.Upsample(
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners if mode in ["bilinear", "bicubic", "trilinear"] else None
        )

    def forward(self, x):
        return self.upscale(x)
    