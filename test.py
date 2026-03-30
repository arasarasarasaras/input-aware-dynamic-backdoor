import torch
from models.generator import TriggerGenerator

# create model
model = TriggerGenerator(inChannels=3, outChannels=3)

# dummy input
x = torch.randn(1, 3, 128, 128)

# forward pass
y = model(x)

print("Output shape:", y.shape)
