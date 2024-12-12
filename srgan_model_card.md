# Model Card for SRGAN

## Model Overview
- **Model Name**: SRGAN (Super-Resolution Generative Adversarial Network)
- **Description**: A GAN-based super-resolution model designed to enhance low-resolution images into high-resolution images by reconstructing fine textures and details.

## Intended Use
- **Primary Use Case**: Improving image resolution for applications in computer vision, such as medical imaging, satellite imaging, and photography.
- **Limitations**: Not suitable for images with extreme artifacts or out-of-distribution features.

## Datasets
- **Training Dataset**: DIV2K dataset
- **Description**: High-resolution and bicubic down-sampled images for training and validation.

## Metrics
- Evaluated using:
  - Signal-to-Noise Ratio (SNR)
  - Peak Signal-to-Noise Ratio (PSNR)
  - Structural Similarity Index Measure (SSIM)
  - Learned Perceptual Image Patch Similarity (LPIPS)

## Model Limitations
- Performance may degrade on noisy or unseen data.
- Training instability may result in artifacts for certain input types.

## Training Configuration
- **Hardware**: NVIDIA RTX 4080
- **Hyperparameters**:
  - Learning Rate: 2e-4 / 2e-4
  - Batch Size: 32
  - Epochs: 100

## Ethical Considerations
- Ensure that the model is not misused to create deceptive or misleading high-resolution images.

## Citations
- [SRGAN Paper](https://arxiv.org/abs/1609.04802)
