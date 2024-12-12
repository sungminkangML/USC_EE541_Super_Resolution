import argparse
import os
import torch
from models.srcnn import SRCNN
from models.srgan import Generator, Discriminator
from models.ngramswin import ngramswin
from load_dataset import get_dataloaders
from train import train_srcnn, train_srgan_generator, train_srgan, train_swinir
from test import test, test_bicubic
from datetime import datetime
from utils import visualize_results


def main(args):
    # Set dataset paths
    train_hr_folder = f"{args.dataset_path}/DIV2K_train_HR"
    train_lr_folder = f"{args.dataset_path}/DIV2K_train_LR_bicubic/{args.ds_rate.upper()}"
    valid_hr_folder = f"{args.dataset_path}/DIV2K_valid_HR"
    valid_lr_folder = f"{args.dataset_path}/DIV2K_valid_LR_bicubic/{args.ds_rate.upper()}"
    
    # Configure device (CPU or GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    use_imagenet_norm = False

    # Select model
    if args.model == "srcnn":
        model = SRCNN()
    elif args.model == "srgan":
        generator = Generator()
        discriminator = Discriminator()
        model = Generator()
        use_imagenet_norm = True
    elif args.model == "srgan_generator":
        model = Generator()
    elif args.model == "bicubic":
        pass
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    print(f"Using model: {args.model}")
    
    # Initialize data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_hr_folder=train_hr_folder,
        train_lr_folder=train_lr_folder,
        val_hr_folder=valid_hr_folder,
        val_lr_folder=valid_lr_folder,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_imagenet_norm=use_imagenet_norm
    )
    
    # Map training functions to models
    train_functions = {
        "srcnn": train_srcnn,
        "srgan_generator": train_srgan_generator,
        "srgan": train_srgan,
        "swinir": train_swinir
    }

    # Training phase
    if args.mode == "train":
        now = datetime.now()
        date = now.strftime("%m-%d_%H-%M")
        model_path = args.save_path + f"{args.model}_{args.ds_rate}_batch{args.batch_size}_epoch{args.epochs}_lr{args.learning_rate}_{date}.pth"
        print("\nStarting training...")

        # Get the appropriate training function
        train_function = train_functions.get(args.model)
        if train_function is None:
            raise ValueError(f"Unsupported model: {args.model}")

        # Train model
        if args.model == "srgan":
            # SRGAN requires both generator and discriminator
            train_function(
                generator=generator,
                discriminator=discriminator,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                save_path=model_path
            )
        else:
            # Train other models
            train_function(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                save_path=model_path
            )
    
    # Testing phase
    elif args.mode == "test":
        if args.model == "bicubic":
            # Evaluate bicubic interpolation
            test_bicubic(device=device, test_loader=test_loader)
        else:
            # Load model and test
            model_path = os.path.join(args.save_path, args.load_path)
            test(model=model, model_path=model_path, device=device, test_loader=test_loader)
            visualize_results(
                model, 
                test_loader, 
                device, 
                num_samples=5, 
                title=f'{args.model}_{args.ds_rate}_batch{args.batch_size}_epoch{args.epochs}_lr{args.learning_rate}'
            )

if __name__ == "__main__":
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Super Resolution Training")
    parser.add_argument("--model", type=str, choices=["srcnn", "srgan", "srgan_generator", "ngramswin", "bicubic"], required=True, help="Model to use: srcnn, srgan, swinir, ngramswin")
    parser.add_argument("--dataset_path", type=str, default="./dataset", help="Path to the dataset directory")
    parser.add_argument("--ds_rate", type=str, choices=["x2", "x3", "x4"], required=True, help="Downscaling rate (x2, x3, x4)")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Mode: train or test")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for DataLoader")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="./results/models/", help="Path to save model")
    parser.add_argument("--load_path", type=str, default="./results/models/", help="Path to load model")
    args = parser.parse_args()

    main(args)
