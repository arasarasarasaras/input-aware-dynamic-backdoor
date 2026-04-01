import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, DownSampleBlock, UpSampleBlock


class MaskGenerator(nn.Module):
    """
    Learned mask generator for the Input-Aware Dynamic Backdoor Attack.

    Same encoder-decoder architecture as TriggerGenerator, but:
      - Outputs 1 channel (grayscale mask) instead of 3 (RGB pattern)
      - Includes a threshold() method to sharpen masks toward binary values
      - Sigmoid output ensures values in [0, 1]

    Pre-trained with diversity loss + norm loss before joint training.
    """

    def __init__(self, inChannels=3):
        super().__init__()

        self.enc1 = nn.Sequential(
            ConvBlock(inChannels, 32),
            ConvBlock(32, 32),
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

        # Final layer: 1 channel output with sigmoid
        self.out = ConvBlock(32, 1, final=True)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)

        x = self.out(x)
        return x

    def threshold(self, x):
        """
        Sharpen mask values toward 0 or 1 using a steep tanh.

        tanh(10 * (x - 0.5)) maps:
          - values near 0   -> ~-1  -> (after rescaling) ~0
          - values near 0.5 -> ~0   -> 0.5
          - values near 1   -> ~1   -> ~1

        This encourages near-binary masks without hard thresholding,
        keeping gradients flowing during training.
        """
        return (torch.tanh(10.0 * (x - 0.5)) + 1.0) / 2.0