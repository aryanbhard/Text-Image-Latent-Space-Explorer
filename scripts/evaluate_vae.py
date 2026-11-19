"""
Evaluation script for the trained Fashion-MNIST 2D Variational Autoencoder.
Computes test reconstruction loss, KL divergence, and latent coordinate decoding sanity checks.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.vae import FashionVAE
from utils.config import VAE_WEIGHTS_PATH, ARTIFACTS_DIR, DATA_DIR, VAE_LATENT_DIM


def evaluate_vae(weights_path=VAE_WEIGHTS_PATH) -> dict:
    device = torch.device("cpu")
    model = FashionVAE(latent_dim=VAE_LATENT_DIM).to(device)

    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}. Run train_vae.py first.")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.FashionMNIST(root=str(DATA_DIR), train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    total_loss, total_bce, total_kld = 0.0, 0.0, 0.0
    with torch.no_grad():
        for data, _ in test_loader:
            recon, mu, logvar, _ = model(data)
            loss_dict = FashionVAE.loss_function(recon, data, mu, logvar)
            total_loss += loss_dict["loss"].item() * len(data)
            total_bce += loss_dict["bce"].item() * len(data)
            total_kld += loss_dict["kld"].item() * len(data)

    n_samples = len(test_dataset)
    metrics = {
        "test_samples": n_samples,
        "mean_elbo_loss": round(total_loss / n_samples, 4),
        "mean_bce_reconstruction_loss": round(total_bce / n_samples, 4),
        "mean_kl_divergence": round(total_kld / n_samples, 4),
        "latent_dimension": VAE_LATENT_DIM,
        "model_architecture": "ConvVAE-32-64-FC128-Latent2",
    }

    # Test coordinate decoding
    test_coords = torch.tensor([[0.0, 0.0], [1.5, -1.5]], dtype=torch.float32)
    with torch.no_grad():
        decoded = model.decode(test_coords)
    metrics["decoding_check_passed"] = bool(
        decoded.shape == (2, 1, 28, 28) and (0.0 <= decoded.min() <= decoded.max() <= 1.0)
    )

    out_path = ARTIFACTS_DIR / "vae_evaluation_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[VAE Evaluation] Results:", json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate_vae()
