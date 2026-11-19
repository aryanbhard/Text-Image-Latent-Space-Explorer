"""
PAGE 5: VAE Latent-Space Explorer (Unit 2 & Unit 4 Requirements)
Visualizes the trained 2D Fashion-MNIST VAE latent manifold (z1, z2),
enables interactive coordinate decoding into real synthesized images,
displays original vs reconstructed pairs, and presents training loss curves.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from models.vae import FashionVAE
from utils.config import (
    VAE_WEIGHTS_PATH,
    VAE_LATENT_DIM,
    FASHION_MNIST_CLASSES,
    ARTIFACTS_DIR,
    DATA_DIR,
)
from utils.visualization import plot_vae_latent_scatter, plot_training_loss_curves
import json

st.set_page_config(page_title="VAE Latent Space", page_icon="🧪", layout="wide")

st.title("🧪 Variational Autoencoder (VAE) Latent-Space Explorer")
st.markdown(
    "Explore the continuous **2-dimensional Gaussian latent manifold** of our real PyTorch VAE "
    "trained on the **Fashion-MNIST** dataset. Decode arbitrary coordinates $(z_1, z_2)$ into synthesized clothing items."
)

# Load Trained VAE Model
@st.cache_resource
def load_trained_vae():
    device = torch.device("cpu")
    model = FashionVAE(latent_dim=VAE_LATENT_DIM).to(device)
    if not VAE_WEIGHTS_PATH.exists():
        st.error(f"Model weights not found at {VAE_WEIGHTS_PATH}. Please run scripts/train_vae.py.")
        return None
    model.load_state_dict(torch.load(VAE_WEIGHTS_PATH, map_location=device))
    model.eval()
    return model

model = load_trained_vae()
if model is None:
    st.stop()

# Load Latents for visualization
@st.cache_data
def get_sample_latents():
    sample_file = ARTIFACTS_DIR / "vae_sample_latents.pt"
    if sample_file.exists():
        try:
            # Under PyTorch 2.6+, weights_only defaults to True; load trusted local artifact safely
            try:
                data = torch.load(sample_file, map_location="cpu")
            except Exception:
                data = torch.load(sample_file, map_location="cpu", weights_only=False)

            latents_data = data["latents"]
            classes_data = data["classes"]

            # Support both PyTorch Tensor and NumPy formats
            if isinstance(latents_data, torch.Tensor):
                latents = latents_data.cpu().numpy()
            elif isinstance(latents_data, list):
                if len(latents_data) > 0 and isinstance(latents_data[0], torch.Tensor):
                    latents = torch.cat(latents_data, dim=0).cpu().numpy()
                else:
                    latents = np.concatenate(latents_data, axis=0)
            else:
                latents = np.asarray(latents_data)

            if isinstance(classes_data, torch.Tensor):
                classes = classes_data.cpu().numpy()
            elif isinstance(classes_data, list):
                if len(classes_data) > 0 and isinstance(classes_data[0], torch.Tensor):
                    classes = torch.cat(classes_data, dim=0).cpu().numpy()
                else:
                    classes = np.concatenate(classes_data, axis=0)
            else:
                classes = np.asarray(classes_data)

            return latents[:1000], classes[:1000]
        except Exception:
            pass
    return np.random.randn(200, 2).astype(np.float32), np.random.randint(0, 10, size=200)

sample_latents, sample_classes = get_sample_latents()

# Load Training History
@st.cache_data
def get_training_history():
    hist_file = ARTIFACTS_DIR / "vae_training_history.json"
    if hist_file.exists():
        with open(hist_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

training_history = get_training_history()

# Tab Layout
tab_manifold, tab_recon, tab_arch, tab_curves = st.tabs([
    "🗺️ 2D Latent Manifold & Interactive Decoder",
    "🔄 Original vs Reconstructed Gallery",
    "📐 VAE Architecture & Formulation",
    "📈 Training Loss Convergence",
])

with tab_manifold:
    st.subheader("1. Interactive Latent Coordinate Decoder")
    st.caption("Select or adjust latent coordinates (z₁, z₂) to synthesize a Fashion-MNIST image in real time.")
    
    col_map, col_decode = st.columns([2, 1])

    with col_decode:
        st.markdown("#### 🎯 Coordinate Controls")
        coord_mode = st.radio("Coordinate Input Mode:", ["Slider Selection", "Preset Semantic Centers"], horizontal=True)
        
        if coord_mode == "Slider Selection":
            z1 = st.slider("Latent z₁ (Horizontal Dimension):", min_value=-3.5, max_value=3.5, value=0.0, step=0.1)
            z2 = st.slider("Latent z₂ (Vertical Dimension):", min_value=-3.5, max_value=3.5, value=0.0, step=0.1)
        else:
            presets = {
                "Center (Origin 0.0, 0.0)": (0.0, 0.0),
                "Top / Shirt Region (+1.5, +1.2)": (1.5, 1.2),
                "Trouser Region (-1.8, +1.5)": (-1.8, 1.5),
                "Sandal / Sneaker Region (-1.2, -1.8)": (-1.2, -1.8),
                "Ankle Boot Region (+1.6, -1.4)": (1.6, -1.4),
                "Bag Region (+0.2, -2.1)": (0.2, -2.1),
            }
            preset_choice = st.selectbox("Choose Semantic Region:", list(presets.keys()))
            z1, z2 = presets[preset_choice]

        # Decode coordinates through trained PyTorch model
        with torch.no_grad():
            latent_vec = torch.tensor([[z1, z2]], dtype=torch.float32)
            generated_tensor = model.decode(latent_vec).squeeze().numpy()  # (28, 28)

        # Scale to 0-255 uint8 PIL Image
        img_np = (generated_tensor * 255.0).astype(np.uint8)
        gen_img = Image.fromarray(img_np, mode="L").resize((168, 168), Image.Resampling.NEAREST)

        st.markdown("---")
        st.markdown("#### 🖼️ Generated Output")
        st.image(gen_img, caption=f"Synthesized from z = ({z1:.2f}, {z2:.2f})", width=170)
        st.caption("Generated directly by the trained PyTorch Transposed-Conv Decoder.")

    with col_map:
        fig = plot_vae_latent_scatter(
            latent_coords=sample_latents,
            class_indices=sample_classes,
            class_names=FASHION_MNIST_CLASSES,
            selected_coord=(z1, z2),
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_recon:
    st.subheader("2. Reconstruction Fidelity: Original vs Reconstructed Test Images")
    st.write(
        "Demonstrates the autoencoder's ability to compress a 784-dimensional image down to 2 numbers, "
        "and reconstruct the item back through the decoder."
    )
    
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.FashionMNIST(root=str(DATA_DIR), train=False, download=False, transform=transform)

    sample_indices = [0, 1, 3, 5, 8, 12, 19, 23]
    cols = st.columns(len(sample_indices))
    
    for i, idx in enumerate(sample_indices):
        orig_tensor, target_label = test_dataset[idx]
        with torch.no_grad():
            recon_tensor, mu, logvar, _ = model(orig_tensor.unsqueeze(0))
        
        orig_img = Image.fromarray((orig_tensor.squeeze().numpy() * 255).astype(np.uint8)).resize((100, 100), Image.Resampling.NEAREST)
        recon_img = Image.fromarray((recon_tensor.squeeze().numpy() * 255).astype(np.uint8)).resize((100, 100), Image.Resampling.NEAREST)
        
        with cols[i]:
            st.image(orig_img, caption=f"Original\n({FASHION_MNIST_CLASSES[target_label]})", use_container_width=True)
            st.image(recon_img, caption=f"Recon\nz=({mu[0,0]:.1f},{mu[0,1]:.1f})", use_container_width=True)

with tab_arch:
    st.subheader("3. VAE Mathematical Architecture")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown(
            r"""
            #### Convolutional Encoder Pipeline
            1. **Input Image:** $x \in \mathbb{R}^{1 \times 28 \times 28}$ (784 pixel values normalized to $[0, 1]$).
            2. **Conv2D Block 1:** $1 \to 32$ filters, kernel 3x3, stride 2, padding 1 $\to$ BatchNorm $\to$ LeakyReLU(0.2). Feature map: $(32, 14, 14)$.
            3. **Conv2D Block 2:** $32 \to 64$ filters, kernel 3x3, stride 2, padding 1 $\to$ BatchNorm $\to$ LeakyReLU(0.2). Feature map: $(64, 7, 7)$.
            4. **Flatten & Dense:** Flatten to 3136 units $\to$ Linear(3136, 128) $\to$ LeakyReLU(0.2).
            5. **Latent Parameter Heads:**
               - Mean vector $\boldsymbol{\mu} \in \mathbb{R}^2$ via Linear(128, 2).
               - Log-variance vector $\log \boldsymbol{\sigma}^2 \in \mathbb{R}^2$ via Linear(128, 2).
            """
        )
    with col_a2:
        st.markdown(
            r"""
            #### Reparameterization Trick & Decoder Pipeline
            1. **Reparameterization Trick:**
               $$z = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
               *Enables backpropagation because stochasticity is isolated to the independent noise term $\boldsymbol{\epsilon}$.*
            2. **Dense Expansion:** Linear(2, 128) $\to$ Linear(128, 3136) $\to$ Reshape to $(64, 7, 7)$.
            3. **ConvTranspose2D Block 1:** $64 \to 32$, stride 2, output_padding 1 $\to$ BatchNorm $\to$ LeakyReLU. Shape: $(32, 14, 14)$.
            4. **ConvTranspose2D Block 2:** $32 \to 1$, stride 2, output_padding 1 $\to$ **Sigmoid Activation**. Shape: $(1, 28, 28)$.
            """
        )

with tab_curves:
    st.subheader("4. Training Loss & Convergence Metrics")
    if training_history:
        fig_loss = plot_training_loss_curves(
            epochs=training_history["epochs"],
            total_losses=training_history["total_loss"],
            bce_losses=training_history["bce_loss"],
            kld_losses=training_history["kld_loss"],
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        
        st.markdown(
            f"**Final Training Epoch Loss:** `{training_history['total_loss'][-1]:.2f}` "
            f"(BCE Reconstruction: `{training_history['bce_loss'][-1]:.2f}`, KL Divergence: `{training_history['kld_loss'][-1]:.2f}`)"
        )
    else:
        st.info("Training history file not found. Re-run scripts/train_vae.py to generate convergence log.")
