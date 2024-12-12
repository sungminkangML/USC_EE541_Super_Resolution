import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import lpips
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def compute_snr(hr_image, sr_image):
    """
    Compute Signal-to-Noise Ratio (SNR).
    Args:
        hr_image (numpy array): High-resolution ground truth image.
        sr_image (numpy array): Super-resolved image.
    Returns:
        float: SNR value.
    """
    signal_power = np.sum(hr_image ** 2)
    noise_power = np.sum((hr_image - sr_image) ** 2)
    return float('inf') if noise_power == 0 else 10 * np.log10(signal_power / noise_power)


def compute_psnr(hr_image, sr_image):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    Args:
        hr_image (numpy array): High-resolution ground truth image.
        sr_image (numpy array): Super-resolved image.
    Returns:
        float: PSNR value.
    """
    return compare_psnr(hr_image, sr_image, data_range=hr_image.max() - hr_image.min())


def compute_ssim(hr_image, sr_image):
    """
    Compute Structural Similarity Index Measure (SSIM).
    Args:
        hr_image (numpy array): High-resolution ground truth image.
        sr_image (numpy array): Super-resolved image.
    Returns:
        float: SSIM value.
    """
    min_dim = min(hr_image.shape[0], hr_image.shape[1])
    win_size = min(7, max(3, min_dim // 2 * 2 - 1))
    data_range = hr_image.max() - hr_image.min()
    return compare_ssim(hr_image, sr_image, win_size=win_size, data_range=data_range, channel_axis=-1)


def compute_lpips(hr_image, sr_image, lpips_net='vgg'):
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS).
    Args:
        hr_image (numpy array or torch.Tensor): High-resolution ground truth image.
        sr_image (numpy array or torch.Tensor): Super-resolved image.
        lpips_net (str): Backbone network for LPIPS calculation ('vgg' or 'alex').
    Returns:
        float: LPIPS value.
    """
    if isinstance(hr_image, np.ndarray):
        hr_image = torch.tensor(hr_image, dtype=torch.float32).permute(2, 0, 1)
    if isinstance(sr_image, np.ndarray):
        sr_image = torch.tensor(sr_image, dtype=torch.float32).permute(2, 0, 1)

    hr_image = (hr_image.unsqueeze(0).float() / 255.0 * 2) - 1
    sr_image = (sr_image.unsqueeze(0).float() / 255.0 * 2) - 1

    lpips_model = lpips.LPIPS(net=lpips_net).to(hr_image.device)
    lpips_model.eval()
    lpips_value = lpips_model(hr_image, sr_image)
    return lpips_value.item()


def plot_metrics(epochs, train_losses, val_losses, train_psnrs, val_psnrs, train_snrs, val_snrs, train_ssims, val_ssims, output_path):
    """
    Plot training and validation metrics (Loss, PSNR, SNR, SSIM) and save as an image.
    Args:
        epochs (list): List of epoch numbers.
        train_losses, val_losses (list): Loss values.
        train_psnrs, val_psnrs (list): PSNR values.
        train_snrs, val_snrs (list): SNR values.
        train_ssims, val_ssims (list): SSIM values.
        output_path (str): Path to save the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = [
        (train_losses, val_losses, "Loss", "Loss"),
        (train_psnrs, val_psnrs, "PSNR", "PSNR (dB)"),
        (train_snrs, val_snrs, "SNR", "SNR (dB)"),
        (train_ssims, val_ssims, "SSIM", "SSIM")
    ]

    for ax, (train, val, title, ylabel) in zip(axes.ravel(), metrics):
        ax.plot(epochs, train, label="Train")
        ax.plot(epochs, val, label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(ylabel)
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def visualize_results(model, test_loader, device, num_samples=5, title=''):
    """
    Visualize low-resolution, super-resolved, and high-resolution images.
    Args:
        model (torch.nn.Module): Super-resolution model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device for inference.
        num_samples (int): Number of samples to visualize.
        title (str): Title for the plots.
    """
    model.eval()
    samples_shown = 0

    for lr_image, hr_image in test_loader:
        lr_image, hr_image = lr_image.to(device), hr_image.to(device)
        with torch.no_grad():
            sr_image = model(lr_image, target_size=(hr_image.shape[2], hr_image.shape[3]))

        lr_image_np = lr_image.cpu().numpy().transpose(0, 2, 3, 1)
        sr_image_np = sr_image.cpu().numpy().transpose(0, 2, 3, 1)
        hr_image_np = hr_image.cpu().numpy().transpose(0, 2, 3, 1)

        for i in range(lr_image_np.shape[0]):
            plt.figure(figsize=(12, 4))
            plt.suptitle(title, fontsize=16)
            plt.subplot(1, 3, 1)
            plt.title("Low-Resolution Input")
            plt.imshow(np.clip(lr_image_np[i], 0, 1))
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.title("Super-Resolved Output")
            plt.imshow(np.clip(sr_image_np[i], 0, 1))
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.title("High-Resolution Ground Truth")
            plt.imshow(np.clip(hr_image_np[i], 0, 1))
            plt.axis('off')
            plt.show()

            samples_shown += 1
            if samples_shown >= num_samples:
                return


def bicubic_interpolation(lr_image, scale_factor):
    """
    Perform bicubic interpolation on a low-resolution image.
    Args:
        lr_image (numpy array): Low-resolution input image (HWC format).
        scale_factor (int): Upscaling factor.
    Returns:
        numpy array: Upscaled image using bicubic interpolation (HWC format).
    """
    return cv2.resize(lr_image, (lr_image.shape[1] * scale_factor, lr_image.shape[0] * scale_factor), interpolation=cv2.INTER_CUBIC)


def evaluate_bicubic(lr_images, hr_images, scale_factor):
    """
    Evaluate bicubic interpolation with PSNR and SSIM.
    Args:
        lr_images (list): Low-resolution images.
        hr_images (list): High-resolution ground truth images.
        scale_factor (int): Upscaling factor.
    Returns:
        dict: Average PSNR and SSIM.
    """
    psnr_values = []
    ssim_values = []

    for lr, hr in zip(lr_images, hr_images):
        sr = bicubic_interpolation(lr, scale_factor)
        psnr_values.append(compute_psnr(hr, sr))
        ssim_values.append(compute_ssim(hr, sr))

    return {
        "avg_psnr": np.mean(psnr_values) if psnr_values else 0.0,
        "avg_ssim": np.mean(ssim_values) if ssim_values else 0.0
    }
