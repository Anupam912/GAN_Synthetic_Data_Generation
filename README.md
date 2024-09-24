# **Advanced GAN Project**

This project implements a state-of-the-art **Generative Adversarial Network (GAN)** using several advanced techniques like **WGAN-GP**, **Self-Attention**, **Conditional GAN (cGAN)** with multi-attribute control, **Adaptive Discriminator Augmentation (ADA)**, and **Exponential Moving Average (EMA)** for generating high-quality images. The project is built using **PyTorch** and can be extended to handle various datasets.

## **Key Features**

- **WGAN-GP** (Wasserstein GAN with Gradient Penalty) for stable training.
- **cGAN** with multi-attribute control for conditional image generation.
- **Self-Attention** for capturing long-range dependencies in image generation.
- **Adaptive Discriminator Augmentation (ADA)** to prevent overfitting.
- **Spectral Normalization** for both the generator and the discriminator.
- **Exponential Moving Average (EMA)** for stable generator updates.
- **Mixed Precision Training (FP16)** for faster training and reduced memory usage.

## **Table of Contents**

- [Requirements](#requirements)
- [Setup](#setup)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Generate Images](#generate-images)
- [Loading Pre-trained Models](#loading-pre-trained-models)
- [Evaluation and Metrics](#evaluation-and-metrics)

## **Requirements**

Make sure you have the following libraries installed in your environment:

```bash
pip install torch torchvision tqdm numpy pillow tensorboard
```

## **Setup**

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/gan-project.git
cd gan-project
```

Create the necessary directories:

```bash
mkdir checkpoints generated_images
```

## **Dataset**

1. Download the img_align_celeba.zip file.
2. Extract the images to a ./data/celeba/ folder.

## **Training the Model**

```bash
python main.py
```

### **Hyperparameters**

You can modify the hyperparameters in main.py:

- latent_dim: Size of the latent space (default: 100).
- batch_size: Number of images processed per batch (default: 64).
- n_epochs: Number of epochs for training (default: 200).
- lr: Learning rate for optimizers (default: 0.0002).
- p_augmentation: Probability of applying data augmentation during ADA (default: 0.1).

## **Generate Images**

After training, you can generate new images by loading the trained EMA generator model and feeding random noise vectors. The generated images will be saved to the generated_images/ folder.

```python
from model import GeneratorWithSpectralNorm
import torch
from torchvision.utils import save_image

# Load the generator model
generator = GeneratorWithSpectralNorm(latent_dim=100, num_classes=10).to('cuda')
generator.load_state_dict(torch.load('checkpoints/generator_epoch_199.pth'))
generator.eval()

# Generate images
fixed_noise = torch.randn(64, 100, 1, 1).to('cuda')
with torch.no_grad():
    fake_images = generator(fixed_noise, labels)  # Labels for cGAN can be passed here
    save_image(fake_images, "generated_images/final_output.png", normalize=True)
```

## **Loading Pre-trained Models**

To resume training from a checkpoint or to use a pre-trained model, load the saved checkpoint like this:

```python
generator.load_state_dict(torch.load('checkpoints/generator_epoch_199.pth'))
discriminator.load_state_dict(torch.load('checkpoints/discriminator_epoch_199.pth'))
```

## **Evaluation and Metrics**

### Qualitative Evaluation

Generated images are saved in the generated_images/ directory. You can visualize them directly to assess the quality of the results.

### Quantitative Evaluation

You can evaluate the model using standard GAN evaluation metrics like Inception Score (IS) or Fréchet Inception Distance (FID). These metrics help evaluate the diversity and quality of generated images compared to real images.

```bash
pip install pytorch-fid
python -m pytorch_fid data/celeba generated_images/ #For FID
```
