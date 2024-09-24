import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

# Self-Attention Layer for SAGAN
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        attention = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = nn.Softmax(dim=-1)(attention)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

# Generator with Spectral Normalization
class GeneratorWithSpectralNorm(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(GeneratorWithSpectralNorm, self).__init__()
        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.model = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False)),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output between -1 and 1
        )

        # Adding self-attention at the final layers
        self.attention = SelfAttention(128)

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        gen_input = gen_input.view(gen_input.size(0), gen_input.size(1), 1, 1)
        img = self.model(gen_input)
        img = self.attention(img)  # Apply self-attention
        return img

# Multi-Scale Discriminator with Spectral Normalization and Auxiliary Classifier (for ACGAN)
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            self._make_discriminator_block(),
            self._make_discriminator_block(),
            self._make_discriminator_block()
        ])
        self.aux_classifier = nn.Linear(512, num_classes)  # Predict attributes

    def _make_discriminator_block(self):
        return nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False))
        )

    def forward(self, img):
        img_half = nn.functional.interpolate(img, scale_factor=0.5)
        img_quarter = nn.functional.interpolate(img, scale_factor=0.25)

        validity_full = self.discriminators[0](img)
        validity_half = self.discriminators[1](img_half)
        validity_quarter = self.discriminators[2](img_quarter)

        # Combine outputs from multiple scales
        validity = validity_full + validity_half + validity_quarter

        # Auxiliary classification output
        features = validity.view(validity.size(0), -1)
        attribute_pred = self.aux_classifier(features)

        return validity, attribute_pred

# Gradient Penalty Function for WGAN-GP
def gradient_penalty(discriminator, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(real_data.device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    d_interpolates = discriminator(interpolates)
    fake = torch.ones(d_interpolates[0].size()).to(real_data.device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates[0], inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
