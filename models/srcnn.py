import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # Feature extraction layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)  # Padding ensures output size is same as input size
        self.relu1 = nn.ReLU(inplace=True)
        
        # Non-linear mapping layer
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Reconstruction layer
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x, target_size=None):
        if target_size:
         # Validate target_size
            if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
                raise ValueError(f"Invalid target_size: {target_size}")
            x = F.interpolate(x, size=target_size, mode='bicubic', align_corners=False)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

