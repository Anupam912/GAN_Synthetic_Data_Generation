import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from torchvision import transforms

# Adaptive Augmentation for ADA
def adaptive_augmentation(real_imgs, p_augmentation):
    if torch.rand(1).item() < p_augmentation:
        real_imgs = transforms.RandomHorizontalFlip()(real_imgs)
        real_imgs = transforms.RandomRotation(15)(real_imgs)
    return real_imgs

# Data augmentation for Adaptive Discriminator Augmentation (ADA)
def adaptive_augmentation(real_imgs, p_augmentation):
    if torch.rand(1).item() < p_augmentation:
        real_imgs = transforms.RandomHorizontalFlip()(real_imgs)
        real_imgs = transforms.RandomRotation(15)(real_imgs)
    return real_imgs

# Function to load CelebA dataset with data augmentation
def get_dataloader(data_dir, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Save generated images
def save_generated_images(generator, fixed_noise, epoch, output_dir="generated_images"):
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()
        save_image(fake_images, f"{output_dir}/epoch_{epoch}.png", nrow=8, normalize=True)
