import torch
from model import GeneratorWithSpectralNorm
from torchvision.utils import save_image

# Load the generator model
latent_dim = 100
num_classes = 10  # Adjust according to your setup

# Initialize generator
generator = GeneratorWithSpectralNorm(latent_dim=latent_dim, num_classes=num_classes).to('cuda')

# Load pre-trained generator weights from a checkpoint
checkpoint_path = 'checkpoints/generator_epoch_199.pth'  # Modify as needed
generator.load_state_dict(torch.load(checkpoint_path))
generator.eval()

# Generate images
fixed_noise = torch.randn(64, latent_dim, 1, 1).to('cuda')
labels = torch.randint(0, num_classes, (64,)).to('cuda')  # Random labels for cGAN

with torch.no_grad():
    generated_images = generator(fixed_noise, labels)
    save_image(generated_images, "generated_images/final_output.png", normalize=True)
