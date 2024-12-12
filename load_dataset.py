import os
import random
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize


class SRDataset(Dataset):
    def __init__(self, hr_image_paths, lr_image_paths, hr_size=(224, 224), lr_size=(112, 112), mean=None, std=None):
        """
        Dataset for loading high-resolution (HR) and low-resolution (LR) image pairs.
        Args:
            hr_image_paths (list): Paths to HR images.
            lr_image_paths (list): Paths to LR images.
            hr_size (tuple): Target dimensions for HR images.
            lr_size (tuple): Target dimensions for LR images.
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.
        """
        self.hr_image_paths = hr_image_paths
        self.lr_image_paths = lr_image_paths
        self.hr_size = hr_size
        self.lr_size = lr_size
        assert len(self.hr_image_paths) == len(self.lr_image_paths), "HR and LR image counts must match."
        self.normalize = Normalize(mean=mean, std=std) if mean and std else None

    def __len__(self):
        return len(self.hr_image_paths)

    def __getitem__(self, idx):
        # Load HR and LR images
        hr_image_path = self.hr_image_paths[idx]
        lr_image_path = self.lr_image_paths[idx]
        hr_image = cv2.imread(hr_image_path)
        lr_image = cv2.imread(lr_image_path)

        # Validate that images are loaded
        if hr_image is None:
            raise ValueError(f"Failed to load HR image at {hr_image_path}")
        if lr_image is None:
            raise ValueError(f"Failed to load LR image at {lr_image_path}")

        # Convert images from BGR to RGB and resize
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.resize(hr_image, self.hr_size, interpolation=cv2.INTER_CUBIC)
        lr_image = cv2.resize(lr_image, self.lr_size, interpolation=cv2.INTER_CUBIC)

        # Convert images to PyTorch tensors and normalize
        hr_image = torch.tensor(hr_image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        lr_image = torch.tensor(lr_image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        if self.normalize:
            hr_image = self.normalize(hr_image)
            lr_image = self.normalize(lr_image)

        return lr_image, hr_image


def calculate_mean_std(dataset):
    """
    Compute mean and standard deviation for LR images in a dataset.
    Args:
        dataset (Dataset): PyTorch Dataset object.
    Returns:
        tuple: (mean, std)
    """
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    mean = torch.zeros(3)
    std = torch.zeros(3)

    for lr_image, _ in loader:  # Only LR images are used
        for channel in range(3):
            mean[channel] += lr_image[:, channel, :, :].mean()
            std[channel] += lr_image[:, channel, :, :].std()

    mean /= len(loader)
    std /= len(loader)

    return mean.tolist(), std.tolist()


def get_dataloaders(train_hr_folder, train_lr_folder, val_hr_folder, val_lr_folder,
                    batch_size=16, num_workers=4, use_imagenet_norm=False):
    """
    Prepare train, validation, and test DataLoaders.
    Args:
        train_hr_folder (str): Directory for HR training images.
        train_lr_folder (str): Directory for LR training images.
        val_hr_folder (str): Directory for HR validation images.
        val_lr_folder (str): Directory for LR validation images.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of data loading workers.
        use_imagenet_norm (bool): Use ImageNet normalization if True.
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load and sort HR and LR image paths
    hr_image_paths = sorted([os.path.join(train_hr_folder, f) for f in os.listdir(train_hr_folder) if f.endswith('.png')])
    lr_image_paths = sorted([os.path.join(train_lr_folder, f) for f in os.listdir(train_lr_folder) if f.endswith('.png')])
    print(f"HR images: {len(hr_image_paths)}, LR images: {len(lr_image_paths)}")
    assert len(hr_image_paths) == len(lr_image_paths), "HR and LR image counts must match."

    # Split data into train and test sets
    indices = list(range(len(hr_image_paths)))
    random.shuffle(indices)
    test_indices = indices[:100]
    train_indices = indices[100:]

    train_hr_paths = [hr_image_paths[i] for i in train_indices]
    train_lr_paths = [lr_image_paths[i] for i in train_indices]
    test_hr_paths = [hr_image_paths[i] for i in test_indices]
    test_lr_paths = [lr_image_paths[i] for i in test_indices]

    # Load validation dataset paths
    val_hr_paths = sorted([os.path.join(val_hr_folder, f) for f in os.listdir(val_hr_folder) if f.endswith('.png')])
    val_lr_paths = sorted([os.path.join(val_lr_folder, f) for f in os.listdir(val_lr_folder) if f.endswith('.png')])

    # Combine train and validation datasets for mean/std calculation
    combined_hr_paths = train_hr_paths + val_hr_paths
    combined_lr_paths = train_lr_paths + val_lr_paths
    combined_dataset = SRDataset(combined_hr_paths, combined_lr_paths)

    # Calculate normalization values
    if use_imagenet_norm:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean, std = calculate_mean_std(combined_dataset)
        print(f"Calculated Mean: {mean}, Std: {std}")

    # Create datasets
    train_dataset = SRDataset(train_hr_paths, train_lr_paths, mean=mean, std=std)
    val_dataset = SRDataset(val_hr_paths, val_lr_paths, mean=mean, std=std)
    test_dataset = SRDataset(test_hr_paths, test_lr_paths, mean=mean, std=std)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Validation Dataset: {len(val_dataset)} samples")
    print(f"Test Dataset: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader
