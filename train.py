import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm
from time import time
from models.srgan import VGGFeatureExtractor
from utils import compute_snr, compute_psnr, compute_ssim, plot_metrics, plot_srgan_metrics
    

def train_srcnn(model, train_loader, val_loader, device, epochs, learning_rate, save_path):
    # Initialize model, optimizer, scheduler, and loss function
    start_time = time()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.7)
    criterion = nn.MSELoss()

    # Metrics storage
    epochs_list = []
    train_losses, val_losses = [], []
    train_psnrs, val_psnrs = [], []
    train_snrs, val_snrs = [], []
    train_ssims, val_ssims = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_psnr, train_snr, train_ssim = 0, 0, 0, 0

        for lr_imgs, hr_imgs in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # Forward pass and loss computation
            preds = model(lr_imgs, target_size=hr_imgs.shape[2:])
            loss = criterion(preds, hr_imgs)

            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate batch-level metrics
            preds_np = preds.detach().cpu().numpy()
            hr_imgs_np = hr_imgs.cpu().numpy()
            for hr, sr in zip(hr_imgs_np, preds_np):
                train_psnr += compute_psnr(hr, sr)
                train_snr += compute_snr(hr, sr)
                train_ssim += compute_ssim(hr, sr)

        # Compute average training metrics
        train_loss /= len(train_loader)
        train_psnr /= len(train_loader.dataset)
        train_snr /= len(train_loader.dataset)
        train_ssim /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss, val_psnr, val_snr, val_ssim = 0, 0, 0, 0

        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                preds = model(lr_imgs, target_size=hr_imgs.shape[2:])
                val_loss += criterion(preds, hr_imgs).item()

                preds_np = preds.cpu().numpy()
                hr_imgs_np = hr_imgs.cpu().numpy()
                for hr, sr in zip(hr_imgs_np, preds_np):
                    val_psnr += compute_psnr(hr, sr)
                    val_snr += compute_snr(hr, sr)
                    val_ssim += compute_ssim(hr, sr)

        # Compute average validation metrics
        val_loss /= len(val_loader)
        val_psnr /= len(val_loader.dataset)
        val_snr /= len(val_loader.dataset)
        val_ssim /= len(val_loader.dataset)

        # Update learning rate scheduler
        scheduler.step()

        # Store metrics for plotting
        epochs_list.append(epoch + 1)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_psnrs.append(val_psnr)
        val_psnrs.append(val_psnr)
        train_snrs.append(train_snr)
        val_snrs.append(val_snr)
        train_ssims.append(train_ssim)
        val_ssims.append(val_ssim)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train PSNR: {train_psnr:.2f}, Val PSNR: {val_psnr:.2f}, "
              f"Train SNR: {train_snr:.2f}, Val SNR: {val_snr:.2f}, "
              f"Train SSIM: {train_ssim:.4f}, Val SSIM: {val_ssim:.4f}, "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # Find best epoch based on combined metric
    val_psnrs, val_ssims = np.array(val_psnrs), np.array(val_ssims)
    normalized_val_psnr = (val_psnrs - val_psnrs.min()) / (val_psnrs.max() - val_psnrs.min() + 1e-6)
    normalized_val_ssim = (val_ssims - val_ssims.min()) / (val_ssims.max() - val_ssims.min() + 1e-6)
    combined_metric = (normalized_val_psnr + normalized_val_ssim) / 2
    best_epoch = np.argmax(combined_metric) + 1

    print(f"\nBest epoch: {best_epoch}, Combined metric: {combined_metric.max():.4f}")
    print(f"Training completed in {time() - start_time:.2f} seconds.")

    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    # Plot training metrics
    plot_path = save_path.replace('/models', '/figures').replace('.pth', '_plot.png')
    plot_metrics(epochs_list, train_losses, val_losses, train_psnrs, val_psnrs, train_snrs, val_snrs, train_ssims, val_ssims, plot_path)


def train_srgan_generator(model, train_loader, val_loader, device, epochs, learning_rate, save_path):
    # Use SRCNN training process for SRGAN generator
    train_srcnn(model, train_loader, val_loader, device, epochs, learning_rate, save_path)


def train_srgan(generator, discriminator, train_loader, val_loader, device, epochs, learning_rate, save_path):
    # Initialize generator, discriminator, and feature extractor
    start_time = time()
    generator.to(device)
    discriminator.to(device)
    vgg = VGGFeatureExtractor(layer="relu5_4", use_input_norm=True).to(device)

    # Optimizers and schedulers
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    g_scheduler = MultiStepLR(g_optimizer, milestones=[1, 2], gamma=0.7)
    d_scheduler = MultiStepLR(d_optimizer, milestones=[1, 2], gamma=0.7)

    # Loss functions
    adversarial_loss = nn.BCELoss()
    content_loss = nn.MSELoss()
    perceptual_loss = nn.MSELoss()

    # Metrics storage
    epochs_list = []
    train_g_losses, train_d_losses = [], []
    train_snrs, val_snrs = [], []
    train_psnrs, val_psnrs = [], []
    train_ssims, val_ssims = [], []

    for epoch in range(epochs):
        # Training phase for generator and discriminator
        generator.train()
        discriminator.train()

        train_g_loss, train_d_loss, train_snr, train_psnr, train_ssim = 0, 0, 0, 0, 0
        for lr_imgs, hr_imgs in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # Train discriminator
            d_optimizer.zero_grad()
            real_outputs = discriminator(hr_imgs)
            d_real_loss = adversarial_loss(real_outputs, torch.ones_like(real_outputs))
            sr_imgs = generator(lr_imgs)
            fake_outputs = discriminator(sr_imgs.detach())
            d_fake_loss = adversarial_loss(fake_outputs, torch.zeros_like(fake_outputs))
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
            train_d_loss += d_loss.item()

            # Train generator
            g_optimizer.zero_grad()
            sr_features = vgg(sr_imgs)
            hr_features = vgg(hr_imgs)
            g_perceptual_loss = perceptual_loss(sr_features, hr_features)
            g_adversarial_loss = adversarial_loss(fake_outputs, torch.ones_like(fake_outputs))
            g_content_loss = content_loss(sr_imgs, hr_imgs)
            g_loss = g_content_loss + 1e-2 * g_adversarial_loss + g_perceptual_loss
            g_loss.backward()
            g_optimizer.step()
            train_g_loss += g_loss.item()

            # Compute batch metrics
            sr_imgs_np = sr_imgs.cpu().numpy()
            hr_imgs_np = hr_imgs.cpu().numpy()
            for hr, sr in zip(hr_imgs_np, sr_imgs_np):
                train_snr += compute_snr(hr, sr)
                train_psnr += compute_psnr(hr, sr)
                train_ssim += compute_ssim(hr, sr)

        # Compute average metrics
        train_g_loss /= len(train_loader)
        train_d_loss /= len(train_loader)
        train_snr /= len(train_loader.dataset)
        train_psnr /= len(train_loader.dataset)
        train_ssim /= len(train_loader.dataset)

        # Validation metrics
        generator.eval()
        val_snr, val_psnr, val_ssim = 0, 0, 0
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                sr_imgs = generator(lr_imgs)
                sr_imgs_np = sr_imgs.cpu().numpy()
                hr_imgs_np = hr_imgs.cpu().numpy()
                for hr, sr in zip(hr_imgs_np, sr_imgs_np):
                    val_snr += compute_snr(hr, sr)
                    val_psnr += compute_psnr(hr, sr)
                    val_ssim += compute_ssim(hr, sr)

        # Average validation metrics
        val_snr /= len(val_loader.dataset)
        val_psnr /= len(val_loader.dataset)
        val_ssim /= len(val_loader.dataset)

        # Update schedulers
        g_scheduler.step()
        d_scheduler.step()

        # Store metrics
        epochs_list.append(epoch + 1)
        train_g_losses.append(train_g_loss)
        train_d_losses.append(train_d_loss)
        train_snrs.append(train_snr)
        val_snrs.append(val_snr)
        train_psnrs.append(train_psnr)
        val_psnrs.append(val_psnr)
        train_ssims.append(train_ssim)
        val_ssims.append(val_ssim)

        print(f"Epoch {epoch + 1}, Generator Loss: {train_g_loss:.4f}, Discriminator Loss: {train_d_loss:.4f}, "
              f"Train PSNR: {train_psnr:.2f}, Val PSNR: {val_psnr:.2f}, "
              f"Train SNR: {train_snr:.2f}, Val SNR: {val_snr:.2f}, "
              f"Train SSIM: {train_ssim:.4f}, Val SSIM: {val_ssim:.4f}")

    # Save generator
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(generator.state_dict(), save_path)
    print(f"Generator model saved at {save_path}")

    # Plot metrics
    plot_path = save_path.replace('/models', '/figures').replace('.pth', '_plot.png')
    plot_srgan_metrics(epochs=epochs_list, train_g_losses=train_g_losses, train_d_losses=train_d_losses,
                       train_snrs=train_snrs, val_snrs=val_snrs, 
                       train_psnrs=train_psnrs, val_psnrs=val_psnrs, 
                       train_ssims=train_ssims, val_ssims=val_ssims, output_path=plot_path)


def train_swinir():
    # Placeholder for SWINIR training
    pass
