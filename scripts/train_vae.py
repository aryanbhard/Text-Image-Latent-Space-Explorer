"""
Training script for Fashion-MNIST 2D Variational Autoencoder (VAE).
Saves trained model weights and convergence metrics for reproducible latent space exploration.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models.vae import FashionVAE
from utils.config import VAE_WEIGHTS_PATH, ARTIFACTS_DIR, DATA_DIR, VAE_LATENT_DIM


def train_fashion_vae(
    epochs: int = 6,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    sample_limit: int = 12000,
    seed: int = 42,
) -> dict:
    """
    Train a 2D VAE on Fashion-MNIST dataset and persist weights & metrics.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    print(f"[VAE Training] Running on device: {device} | Epochs: {epochs} | Latent Dim: {VAE_LATENT_DIM}")

    # Ensure output directories exist
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download and preprocess Fashion-MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),  # Scales images to [0.0, 1.0]
    ])

    print("[VAE Training] Loading Fashion-MNIST dataset...")
    train_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform,
    )

    # Use a clean representative subset for fast, robust CPU training
    if sample_limit and sample_limit < len(train_dataset):
        indices = list(range(sample_limit))
        train_sub = Subset(train_dataset, indices)
    else:
        train_sub = train_dataset

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 2. Instantiate Model and Optimizer
    model = FashionVAE(latent_dim=VAE_LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "epochs": [],
        "total_loss": [],
        "bce_loss": [],
        "kld_loss": [],
        "val_total_loss": [],
        "val_bce_loss": [],
        "val_kld_loss": [],
    }

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_bce = 0.0
        train_kld = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar, _ = model(data)
            loss_dict = FashionVAE.loss_function(recon_batch, data, mu, logvar, beta=1.0)

            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(data)
            train_bce += loss_dict["bce"].item() * len(data)
            train_kld += loss_dict["kld"].item() * len(data)

        n_train = len(train_sub)
        avg_loss = train_loss / n_train
        avg_bce = train_bce / n_train
        avg_kld = train_kld / n_train

        # Validation
        model.eval()
        val_loss, val_bce, val_kld = 0.0, 0.0, 0.0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                recon, mu, logvar, _ = model(data)
                v_dict = FashionVAE.loss_function(recon, data, mu, logvar, beta=1.0)
                val_loss += v_dict["loss"].item() * len(data)
                val_bce += v_dict["bce"].item() * len(data)
                val_kld += v_dict["kld"].item() * len(data)

        n_test = len(test_dataset)
        v_avg_loss = val_loss / n_test
        v_avg_bce = val_bce / n_test
        v_avg_kld = val_kld / n_test

        history["epochs"].append(epoch)
        history["total_loss"].append(round(avg_loss, 4))
        history["bce_loss"].append(round(avg_bce, 4))
        history["kld_loss"].append(round(avg_kld, 4))
        history["val_total_loss"].append(round(v_avg_loss, 4))
        history["val_bce_loss"].append(round(v_avg_bce, 4))
        history["val_kld_loss"].append(round(v_avg_kld, 4))

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {avg_loss:.2f} (BCE: {avg_bce:.2f}, KLD: {avg_kld:.2f}) | "
            f"Val Loss: {v_avg_loss:.2f}"
        )

    # 4. Save trained weights and history
    torch.save(model.state_dict(), str(VAE_WEIGHTS_PATH))
    print(f"[VAE Training] Saved model weights to: {VAE_WEIGHTS_PATH}")

    history_path = ARTIFACTS_DIR / "vae_training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[VAE Training] Saved training history to: {history_path}")

    # Extract sample validation latent coordinates for visualization
    model.eval()
    sample_latents = []
    sample_classes = []
    with torch.no_grad():
        for data, labels in test_loader:
            mu, _ = model.encode(data)
            sample_latents.append(mu.cpu())
            sample_classes.append(labels.cpu())
            if sum(len(l) for l in sample_classes) >= 1000:
                break

    latents_t = torch.cat(sample_latents, dim=0)[:1000]
    classes_t = torch.cat(sample_classes, dim=0)[:1000]
    torch.save({
        "latents": latents_t,
        "classes": classes_t,
    }, ARTIFACTS_DIR / "vae_sample_latents.pt")
    print(f"[VAE Training] Saved validation latent coordinates as pure PyTorch Tensors.")

    return history


if __name__ == "__main__":
    train_fashion_vae(epochs=5, sample_limit=8000)
