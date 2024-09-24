import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from model import GeneratorWithSpectralNorm, MultiScaleDiscriminator, gradient_penalty
from utils import get_dataloader, save_generated_images, adaptive_augmentation
from ema import update_average
from torch.cuda.amp import GradScaler, autocast
import subprocess

# Hyperparameters
latent_dim = 100
num_classes = 10  # Assuming 10 attributes for ACGAN
image_size = 64
batch_size = 64
lr = 0.0002
n_epochs = 200
beta1 = 0.5
lambda_gp = 10  # Gradient penalty weight
n_critic = 5    # Number of discriminator steps per generator step
p_augmentation = 0.1  # Augmentation probability for ADA

# Initialize models
generator = GeneratorWithSpectralNorm(latent_dim, num_classes).to('cuda')
discriminator = MultiScaleDiscriminator(num_classes).to('cuda')

# To load pre-trained models for resuming training
# checkpoint_epoch = 199
# generator.load_state_dict(torch.load(f'checkpoints/generator_epoch_{checkpoint_epoch}.pth'))
# discriminator.load_state_dict(torch.load(f'checkpoints/discriminator_epoch_{checkpoint_epoch}.pth'))

# Initialize optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# EMA for Generator
ema_generator = GeneratorWithSpectralNorm(latent_dim, num_classes).to('cuda')
ema_generator.load_state_dict(generator.state_dict())

# Mixed precision scaler
scaler = GradScaler()

# Fixed noise for generating images during training
fixed_noise = torch.randn(64, latent_dim, 1, 1).to('cuda')

# Data loader
data_dir = './data/celeba'  # Path to CelebA dataset
dataloader = get_dataloader(data_dir, image_size, batch_size)

# Training Loop
for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(tqdm(dataloader)):
        real_imgs = imgs.to('cuda')
        labels = labels.to('cuda')
        batch_size = real_imgs.size(0)

        # Apply adaptive augmentation
        real_imgs = adaptive_augmentation(real_imgs, p_augmentation)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        z = torch.randn(batch_size, latent_dim, 1, 1).to('cuda')

        with autocast():  # Mixed precision for discriminator
            gen_imgs = generator(z, labels)
            real_validity, real_aux = discriminator(real_imgs)
            fake_validity, fake_aux = discriminator(gen_imgs.detach())

            # Gradient penalty for WGAN-GP
            gradient_penalty_value = gradient_penalty(discriminator, real_imgs, gen_imgs)

            # WGAN loss with auxiliary classifier (ACGAN)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty_value

        # Backward and optimize discriminator
        scaler.scale(d_loss).backward()
        scaler.step(optimizer_D)
        scaler.update()

        # -----------------
        #  Train Generator
        # -----------------
        if i % n_critic == 0:
            optimizer_G.zero_grad()

            with autocast():  # Mixed precision for generator
                gen_imgs = generator(z, labels)
                g_loss = -torch.mean(discriminator(gen_imgs)[0])  # Generator loss

            scaler.scale(g_loss).backward()
            scaler.step(optimizer_G)
            scaler.update()

            # Update EMA generator
            update_average(generator, ema_generator)
    # Save generated images from EMA generator
    save_generated_images(ema_generator, fixed_noise, epoch)

    # Optional: Run FID evaluation after every epoch (for example, every 10 epochs)
    if epoch % 10 == 0:
        print("Evaluating FID score...")
        subprocess.run(["python", "-m", "pytorch_fid", "data/celeba", "generated_images/"], check=True)

    # Optionally save model checkpoints
    torch.save(generator.state_dict(), f'checkpoints/generator_epoch_{epoch}.pth')
    torch.save(discriminator.state_dict(), f'checkpoints/discriminator_epoch_{epoch}.pth')
