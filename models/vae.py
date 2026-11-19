"""
Variational Autoencoder (VAE) for Fashion-MNIST with a 2-Dimensional Latent Space.
Implements Encoder, Reparameterization Trick, Decoder, and Loss Functions.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class FashionVAE(nn.Module):
    """
    Convolutional Variational Autoencoder designed for 28x28 grayscale images (Fashion-MNIST).
    Uses a 2D latent space so users can directly explore, visualize, and decode coordinates.
    """

    def __init__(self, latent_dim: int = 2):
        super(FashionVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder Convolutional Backbone
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 14, 14)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 7, 7)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Fully connected projection layers
        self.flatten_dim = 64 * 7 * 7  # 3136
        self.fc_shared = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Mean (mu) and Log-Variance (logvar) heads
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder Backbone
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, self.flatten_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (32, 14, 14)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (1, 28, 28)
            nn.Sigmoid(),  # Bound output to [0, 1] range for image pixels
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode an input image into latent parameters: mu and logvar.
        Args:
            x: Tensor of shape (B, 1, 28, 28)
        Returns:
            mu: Tensor of shape (B, latent_dim)
            logvar: Tensor of shape (B, latent_dim)
        """
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        h = self.fc_shared(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        The Reparameterization Trick:
        Enables backpropagation through stochastic latent sampling:
        z = mu + std * epsilon, where epsilon ~ N(0, I)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During deterministic evaluation, we can directly use the mean
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent coordinates z back into a 28x28 grayscale image.
        Args:
            z: Tensor of shape (B, latent_dim)
        Returns:
            Reconstructed image of shape (B, 1, 28, 28) with pixel values in [0, 1]
        """
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 64, 7, 7)
        return self.decoder_conv(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: Encode -> Sample (Reparameterize) -> Decode
        Returns:
            recon_x: Reconstructed image (B, 1, 28, 28)
            mu: Mean latent vector (B, latent_dim)
            logvar: Log-variance latent vector (B, latent_dim)
            z: Sampled latent coordinates (B, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

    @staticmethod
    def loss_function(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Evidence Lower Bound (ELBO) Loss:
        Loss = Reconstruction Loss (BCE) + Beta * KL Divergence
        """
        # Pixel-wise Binary Cross Entropy summed over dimensions
        bce = F.binary_cross_entropy(recon_x, x, reduction="sum") / x.size(0)

        # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = (
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        )

        total_loss = bce + beta * kld
        return {
            "loss": total_loss,
            "bce": bce,
            "kld": kld,
        }
