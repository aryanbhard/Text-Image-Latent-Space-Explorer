"""
Unit tests for the Fashion-MNIST Variational Autoencoder.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from models.vae import FashionVAE
from utils.config import VAE_WEIGHTS_PATH, VAE_LATENT_DIM


def test_vae_architecture_forward():
    model = FashionVAE(latent_dim=VAE_LATENT_DIM)
    dummy_input = torch.rand(4, 1, 28, 28)

    recon, mu, logvar, z = model(dummy_input)

    assert recon.shape == (4, 1, 28, 28), f"Expected recon shape (4, 1, 28, 28), got {recon.shape}"
    assert mu.shape == (4, VAE_LATENT_DIM), f"Expected mu shape (4, {VAE_LATENT_DIM}), got {mu.shape}"
    assert logvar.shape == (4, VAE_LATENT_DIM), f"Expected logvar shape (4, {VAE_LATENT_DIM}), got {logvar.shape}"
    assert z.shape == (4, VAE_LATENT_DIM), f"Expected z shape (4, {VAE_LATENT_DIM}), got {z.shape}"
    assert 0.0 <= recon.min() <= recon.max() <= 1.0, "Decoded pixels must be sigmoid-bounded in [0, 1]"


def test_vae_loss_calculation():
    dummy_x = torch.rand(4, 1, 28, 28)
    dummy_recon = torch.rand(4, 1, 28, 28)
    dummy_mu = torch.zeros(4, 2)
    dummy_logvar = torch.zeros(4, 2)

    loss_dict = FashionVAE.loss_function(dummy_recon, dummy_x, dummy_mu, dummy_logvar)

    assert "loss" in loss_dict
    assert "bce" in loss_dict
    assert "kld" in loss_dict
    assert loss_dict["loss"].item() > 0.0
    # For mu=0, logvar=0, KL divergence must be 0
    assert abs(loss_dict["kld"].item()) < 1e-4


def test_vae_coordinate_decoding():
    model = FashionVAE(latent_dim=VAE_LATENT_DIM)
    if VAE_WEIGHTS_PATH.exists():
        model.load_state_dict(torch.load(VAE_WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    coords = torch.tensor([[0.0, 0.0], [-1.0, 1.0], [2.0, -2.0]], dtype=torch.float32)
    with torch.no_grad():
        generated_imgs = model.decode(coords)

    assert generated_imgs.shape == (3, 1, 28, 28)
    assert 0.0 <= generated_imgs.min() <= generated_imgs.max() <= 1.0
