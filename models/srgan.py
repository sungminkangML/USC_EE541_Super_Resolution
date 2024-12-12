import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision.transforms import Normalize

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # Residual block with two convolutional layers and a skip connection
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)  # Add skip connection


class Generator(nn.Module):
    def __init__(self, num_residual_blocks=16):
        super(Generator, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )

        # Second convolutional layer to aggregate residual outputs
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # Upsampling layers for increasing resolution
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * (2 ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        # Final output layer
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x, target_size=None):
        initial = self.conv1(x)
        res_out = self.res_blocks(initial)
        res_out = self.conv2(res_out)
        upsampled = self.upsample(initial + res_out)  # Add skip connection
        out = self.conv3(upsampled)

        # Adjust output size if specified
        if target_size is not None:
            h, w = target_size
            out = out[:, :, :h, :w]
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Sequential blocks with convolution, BatchNorm, and LeakyReLU
        self.block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Fully connected layer initialized dynamically based on feature size
        self.fc = None

    def forward(self, x):
        x = self.block(x)
        if self.fc is None:
            # Dynamically create fc layer based on feature map size
            feature_size = x.size(1) * x.size(2) * x.size(3)
            self.fc = nn.Sequential(
                nn.Linear(feature_size, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1024, 1),
                nn.Sigmoid()
            ).to(x.device)  # Ensure correct device placement

        x = x.view(x.size(0), -1)  # Flatten for fc layer
        x = self.fc(x)
        return x


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer="relu5_4", use_input_norm=True):
        """
        Feature extractor using VGG19 for perceptual loss.
        Args:
            layer (str): Layer to extract features from (e.g., 'relu5_4').
            use_input_norm (bool): Whether to normalize input to match ImageNet statistics.
        """
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg19(pretrained=True)
        layer_mapping = {
            "relu1_1": 2, "relu1_2": 4,
            "relu2_1": 7, "relu2_2": 9,
            "relu3_1": 12, "relu3_2": 14,
            "relu3_3": 16, "relu3_4": 18,
            "relu4_1": 21, "relu4_2": 23,
            "relu4_3": 25, "relu4_4": 27,
            "relu5_1": 30, "relu5_2": 32,
            "relu5_3": 34, "relu5_4": 36
        }
        assert layer in layer_mapping, f"Unsupported layer: {layer}. Available layers: {list(layer_mapping.keys())}"
        self.feature_extractor = nn.Sequential(*list(vgg.features[:layer_mapping[layer]]))
        self.use_input_norm = use_input_norm

        # Normalization layer for ImageNet statistics
        if use_input_norm:
            self.normalize = Normalize(
                mean=(0.485, 0.456, 0.406),  # ImageNet mean
                std=(0.229, 0.224, 0.225)   # ImageNet std
            )
        else:
            self.normalize = None

        # Freeze VGG parameters to prevent updates
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = self.normalize(x)  # Normalize input if required
        return self.feature_extractor(x)  # Extract features
