# Model Card for SRCNN

## Model Overview
- **Model Name**: SRCNN (Super-Resolution Convolutional Neural Network)
- **Description**: SRCNN is a pioneering deep learning model for super-resolution, leveraging convolutional neural networks (CNNs) to upscale low-resolution images into high-resolution images. It directly learns the LR-to-HR mapping through end-to-end supervised training.

## Intended Use
- **Primary Use Case**:
  - Enhancing image resolution for applications such as satellite imaging, medical imaging, and general photography.
  - Benchmarking performance of modern SR models against a foundational architecture.
- **Limitations**:
  - Struggles with perceptual quality and texture details compared to modern SR models (e.g., GAN-based approaches).
  - Best suited for scenarios requiring pixel-level accuracy but not necessarily perceptual realism.

## Datasets
- **Training Dataset**: DIV2K dataset
- **Description**: Contains high-resolution (HR) images paired with bicubic down-sampled low-resolution (LR) images at various down-scaling rates (e.g., x2, x3, x4).
- **Data Preprocessing**:
  - Bicubic interpolation was used to upscale LR images to the desired input resolution.
  - Images were normalized and resized to fixed dimensions for consistency.

## Metrics
- Evaluated using the following metrics:
  - **Signal-to-Noise Ratio (SNR)**: Assesses the level of noise in the reconstructed image relative to the original.
  - **Peak Signal-to-Noise Ratio (PSNR)**: Measures pixel-wise similarity between HR and SR images.
  - **Structural Similarity Index Measure (SSIM)**: Evaluates perceptual similarity by comparing structural details, luminance, and contrast.
  - **Learned Perceptual Image Patch Similarity (LPIPS)**: Captures perceptual differences using feature representations from deep neural networks.

## Model Architecture
- SRCNN consists of three main convolutional layers:
  1. **Patch Extraction and Representation**:
     - A \(9 \times 9\) kernel extracts low-level features from the LR image.
  2. **Non-linear Mapping**:
     - A \(5 \times 5\) kernel maps extracted features to a high-level representation.
  3. **Reconstruction**:
     - A \(5 \times 5\) kernel reconstructs the HR image.
- **Loss Function**: Mean Squared Error (MSE) between the predicted HR and ground truth HR images.

## Training Configuration
- **Hardware**: NVIDIA RTX 4080 GPU
- **Hyperparameters**:
  - Learning Rate: 2e-4
  - Batch Size: 32
  - Epochs: 100
- **Learning Rate Scheduler**:
  - StepLR with step size of 13 and decay factor (\(\gamma\)) of 0.7.

## Model Limitations
- Limited ability to handle complex textures and high-frequency details.
- Performs well in pixel-wise metrics like PSNR but struggles with perceptual quality metrics like LPIPS.

## Ethical Considerations
- Ensure that SRCNN is not misused to create deceptive or misleading high-resolution images.
- Evaluate potential biases in the training dataset, particularly when applying the model to specific domains like medical imaging.

## Citations
- Dong, C., Loy, C. C., He, K., & Tang, X. (2015). Image Super-Resolution Using Deep Convolutional Networks. *arXiv preprint arXiv:1501.00092*. [Link](https://arxiv.org/abs/1501.00092)
