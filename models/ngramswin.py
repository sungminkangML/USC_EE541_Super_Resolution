import torch
import torch.nn as nn
import torch.nn.functional as F

class ngramswin(nn.Module):
    def __init__(self, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)):
        super(ngramswin, self).__init__()
        # Example initialization: You can expand this with Swin Transformer details
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        
        # Initial convolution layer for feature extraction
        self.conv1 = nn.Conv2d(3, self.embed_dim, kernel_size=7, stride=2, padding=3)
        self.norm1 = nn.LayerNorm(self.embed_dim)

        # Dummy implementation of Swin Transformer-like blocks
        self.swin_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1, groups=self.embed_dim),  # Depthwise
                nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)  # Pointwise
            )
            for _ in range(sum(self.depths))
        ])
        
        # Output layer
        self.conv_last = nn.Conv2d(self.embed_dim, 3, kernel_size=1)
        
    def forward(self, x, target_size=None):
        if target_size:
            # Validate target_size
            if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
                raise ValueError(f"Invalid target_size: {target_size}")
            x = F.interpolate(x, size=target_size, mode='bicubic', align_corners=False)
        
        # Initial convolution and normalization
        x = self.conv1(x)
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()  # Rearrange for LayerNorm
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # Rearrange back to NCHW

        # Swin Transformer blocks (dummy implementation)
        for swin_block in self.swin_blocks:
            x = swin_block(x) + x  # Add residual connection

        # Output layer
        x = self.conv_last(x)

        # Resize output to match target size
        if target_size:
            x = F.interpolate(x, size=target_size, mode='bicubic', align_corners=False)
        return x