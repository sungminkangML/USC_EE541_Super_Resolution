import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_snr, compute_psnr, compute_ssim, compute_lpips, bicubic_interpolation


def test(model, model_path, device, test_loader):
    # Load the model weights if the path exists
    print("\nStarting testing...")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Set the model to evaluation mode and move it to the selected device
    model.to(device)
    model.eval()
    
    # Initialize total metrics
    total_snr, total_psnr, total_ssim, total_lpips = 0, 0, 0, 0

    # Loop over the test data
    for lr_image, hr_image in test_loader:
        lr_image, hr_image = lr_image.to(device), hr_image.to(device)
        
        with torch.no_grad():
            # Generate super-resolved images
            sr_image = model(lr_image, target_size=(hr_image.shape[2], hr_image.shape[3]))
        
        # Convert tensors to numpy arrays for metric calculations
        sr_image_np = sr_image.detach().cpu().numpy().transpose(0, 2, 3, 1)  # BCHW -> BHWC
        hr_image_np = hr_image.detach().cpu().numpy().transpose(0, 2, 3, 1)  # BCHW -> BHWC

        # Calculate metrics for each image in the batch
        batch_snr, batch_psnr, batch_ssim, batch_lpips = 0, 0, 0, 0
        for i in range(hr_image_np.shape[0]):
            batch_snr += compute_snr(hr_image_np[i], sr_image_np[i])
            batch_psnr += compute_psnr(hr_image_np[i], sr_image_np[i])
            batch_ssim += compute_ssim(hr_image_np[i], sr_image_np[i])
            batch_lpips += compute_lpips(hr_image_np[i], sr_image_np[i])

        # Update total metrics
        total_snr += batch_snr
        total_psnr += batch_psnr
        total_ssim += batch_ssim
        total_lpips += batch_lpips

    # Compute average metrics
    num_images = len(test_loader.dataset)
    mean_snr = total_snr / num_images
    mean_psnr = total_psnr / num_images
    mean_ssim = total_ssim / num_images
    mean_lpips = total_lpips / num_images

    # Print the test results
    print(f"Test values - SNR: {mean_snr:.4f}, PSNR: {mean_psnr:.4f}, SSIM: {mean_ssim:.4f}, LPIPS: {mean_lpips:.4f}")


def test_bicubic(device, test_loader):
    # Evaluate bicubic interpolation
    print("\nStarting bicubic testing...")

    # Collect LR and HR images from the test loader
    lr_images, hr_images = [], []
    for lr_batch, hr_batch in test_loader:
        lr_images.append(lr_batch.numpy())  # Convert to NumPy
        hr_images.append(hr_batch.numpy())

    # Combine batches into arrays
    lr_images = np.concatenate(lr_images, axis=0)
    hr_images = np.concatenate(hr_images, axis=0)

    # Initialize metrics
    scale_factor = 224 // 112
    snr_values, psnr_values, ssim_values, lpips_values = [], [], [], []
    sr_images = []  # Store super-resolved images for visualization

    for i in range(len(lr_images)):
        # Perform bicubic interpolation
        lr_image_hwc = lr_images[i].transpose(1, 2, 0)  # CHW -> HWC
        lr_image_hwc = np.clip(lr_image_hwc * 255.0, 0, 255).astype(np.uint8)
        sr_image_hwc = bicubic_interpolation(lr_image_hwc, scale_factor)
        sr_image_hwc = np.clip(sr_image_hwc, 0, 255).astype(np.uint8)
        sr_images.append(sr_image_hwc.transpose(2, 0, 1))  # HWC -> CHW

        # Convert to tensors for metrics
        sr_image_tensor = torch.tensor(sr_image_hwc.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0) / 255.0
        hr_image_tensor = torch.tensor(hr_images[i], dtype=torch.float32).unsqueeze(0)

        # Calculate metrics
        snr = compute_snr(hr_image_tensor.squeeze(0).permute(1, 2, 0).numpy(), sr_image_hwc / 255.0)
        psnr = compute_psnr(hr_image_tensor.squeeze(0).permute(1, 2, 0).numpy(), sr_image_hwc / 255.0)
        ssim = compute_ssim(hr_image_tensor.squeeze(0).permute(1, 2, 0).numpy(), sr_image_hwc / 255.0)
        lpips_value = compute_lpips(hr_image_tensor.squeeze(0).permute(1, 2, 0).numpy(), sr_image_hwc / 255.0)

        snr_values.append(snr)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lpips_value)

    # Compute average metrics
    avg_snr = np.mean(snr_values)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)

    # Print results
    print(f'Bicubic Test Results - SNR: {avg_snr:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}')

    # Visualize a few results
    num_samples = 5
    for i in range(min(num_samples, len(lr_images))):
        hr_image_hwc = np.clip(hr_images[i].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        sr_image_hwc = np.clip(sr_images[i].transpose(1, 2, 0), 0, 255).astype(np.uint8)
        lr_image_hwc = np.clip(lr_images[i].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)

        # Plot the images
        plt.figure(figsize=(15, 5))
        plt.suptitle("Bicubic Interpolation Result", fontsize=12)

        plt.subplot(1, 3, 1)
        plt.imshow(lr_image_hwc)
        plt.title("Low-Resolution (112x112)")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(sr_image_hwc)
        plt.title("Bicubic (224x224)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(hr_image_hwc)
        plt.title("High-Resolution (224x224)")
        plt.axis("off")

        plt.show()
