import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, DownSampleBlock, UpSampleBlock

class TriggerGenerator(nn.Module):
    def __init__(self, inChannels=3, outChannels=3):
        super().__init__()

        self.enc1 = nn.Sequential(
            ConvBlock(inChannels, 32),
            ConvBlock(32,32),
            DownSampleBlock()
        )
        self.enc2 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            DownSampleBlock()
        )

        self.enc3 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            DownSampleBlock()
        )

        self.dec1 = nn.Sequential(
            ConvBlock(128, 128),
            UpSampleBlock(),
            ConvBlock(128, 128),
        )

        self.dec2 = nn.Sequential(
            ConvBlock(128, 64),
            UpSampleBlock(),
            ConvBlock(64, 64)
        )

        self.dec3 = nn.Sequential(
            ConvBlock(64, 32),
            UpSampleBlock(),
            ConvBlock(32, 32)
        )

        self.out = ConvBlock(32, outChannels, final=True)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)

        x = self.out(x)
        return x