# USC_EE541_Super_Resolution
Final Project for USC EE541 - Super Resolution

This project implements and compares three foundational models in Super-Resolution (SR):
- **SRCNN (Super-Resolution Convolutional Neural Network):** A pioneering CNN-based SR model.
- **SRGAN Generator-Only:** The generator component of SRGAN trained without adversarial loss.
- **SRGAN (Super-Resolution Generative Adversarial Network):** A GAN-based SR model that incorporates adversarial training for perceptually realistic results.

We evaluate these models using the **DIV2K** dataset and analyze their performance with the following metrics:
- **SNR (Signal-to-Noise Ratio)**
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index Measure)**
- **LPIPS (Learned Perceptual Image Patch Similarity)**

---

## Dataset download
In this project, we use the HR and the bicubic down-sampled images from the DIV2K dataset.

### Download code
 Run the following commands in the terminal (or command prompt for Windows) to download the dataset to your desired directory:
 ```bash
 wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip
 wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip
 wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_test_LR_bicubic_X2.zip
 wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
 wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
 wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_test_HR.zip
```


## Run code
You can run the code with the each code below:

### Train code
 ```bash
python main.py --model srcnn --dataset_path ./dataset --ds_rate x4 --mode train --batch_size 32 --epochs 100 --learning_rate 2e-4
```

### Test code
 ```bash
python main.py --model srgan --dataset_path ./dataset --ds_rate x2 --mode test --batch_size 16 --epochs 50 --learning_rate 1e-4 --load_path srgan_x2_batch16_epoch50_lr0.0001_12-08_19-03.pth
```
