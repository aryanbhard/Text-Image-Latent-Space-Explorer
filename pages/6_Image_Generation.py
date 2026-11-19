"""
PAGE 6: Image Generation Studio (Unit 4 Requirements)
Interactive synthesis interface using the trained PyTorch Fashion-MNIST VAE.
Features coordinate sliders, stochastic Gaussian sampling, batch synthesis gallery,
PNG downloads, and generation pipeline explanation.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import io
import streamlit as st
import torch
import numpy as np
from PIL import Image
from models.vae import FashionVAE
from utils.config import VAE_WEIGHTS_PATH, VAE_LATENT_DIM

st.set_page_config(page_title="Image Generation Studio", page_icon="🎨", layout="wide")

st.title("🎨 Image Generation Studio")
st.markdown(
    "Synthesize novel Fashion-MNIST garments in real time using the decoder network of our trained VAE. "
    "Explore controlled coordinate navigation and stochastic latent sampling."
)

# Load Model
@st.cache_resource
def load_vae_model():
    model = FashionVAE(latent_dim=VAE_LATENT_DIM)
    if not VAE_WEIGHTS_PATH.exists():
        st.error(f"Trained weights not found at {VAE_WEIGHTS_PATH}. Run scripts/train_vae.py first.")
        return None
    model.load_state_dict(torch.load(VAE_WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_vae_model()
if model is None:
    st.stop()

# Synthesis Mode
col_ctrl, col_view = st.columns([1, 2])

with col_ctrl:
    st.subheader("1. Generation Controls")
    gen_mode = st.radio("Synthesis Technique:", ["Stochastic Sampling (Prior N(0, I))", "Controlled Latent Sliders"])

    if gen_mode.startswith("Stochastic"):
        num_images = st.slider("Number of Images to Generate:", min_value=1, max_value=12, value=6)
        seed_val = st.number_input("Random Seed:", min_value=0, max_value=99999, value=42, step=1)
        torch.manual_seed(int(seed_val))
        latent_vectors = torch.randn(num_images, VAE_LATENT_DIM)
    else:
        num_images = 1
        z1 = st.slider("Coordinate z₁:", -3.0, 3.0, 0.5, 0.1)
        z2 = st.slider("Coordinate z₂:", -3.0, 3.0, -0.8, 0.1)
        latent_vectors = torch.tensor([[z1, z2]], dtype=torch.float32)

    st.markdown("---")
    st.markdown("#### 🔬 Generation Pipeline")
    st.markdown(
        """
        1. **Latent Vector:** $\mathbf{z} \in \mathbb{R}^2$
        2. **Dense Projection:** $\mathbf{z} \to \text{FC}(128) \to \text{FC}(3136)$
        3. **Spatial Reshape:** $(64, 7, 7)$ feature maps
        4. **Deconvolution 1:** ConvTranspose2D $\to (32, 14, 14)$
        5. **Deconvolution 2:** ConvTranspose2D $\to (1, 28, 28)$
        6. **Sigmoid Activation:** Bounded pixel intensities $[0, 1]$
        """
    )

with col_view:
    st.subheader("2. Synthesized Image Gallery")
    
    # Run PyTorch Decoder Inference
    with torch.no_grad():
        generated_tensors = model.decode(latent_vectors)  # (N, 1, 28, 28)

    grid_cols = st.columns(min(4, num_images))
    for i in range(num_images):
        raw_img = (generated_tensors[i].squeeze().numpy() * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(raw_img, mode="L").resize((140, 140), Image.Resampling.NEAREST)
        
        # Prepare PNG buffer for download
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        z_coords_str = f"z=({latent_vectors[i,0]:.2f}, {latent_vectors[i,1]:.2f})"
        
        with grid_cols[i % len(grid_cols)]:
            st.image(pil_img, caption=f"Sample #{i+1}\n{z_coords_str}", use_container_width=True)
            st.download_button(
                label=f"💾 Download #{i+1}",
                data=byte_im,
                file_name=f"vae_generated_{i+1}.png",
                mime="image/png",
                key=f"dl_btn_{i}",
            )

st.markdown("---")
st.subheader("3. Latent Space Traversal (Walk Along Manifold)")
st.caption("Inspect smooth morphological transitions by interpolating across the continuous 2D latent space.")

steps = st.slider("Interpolation Steps:", min_value=5, max_value=10, value=7)
col_start, col_end = st.columns(2)
with col_start:
    start_z1 = st.slider("Start z₁:", -3.0, 3.0, -2.0, 0.5)
    start_z2 = st.slider("Start z₂:", -3.0, 3.0, -1.5, 0.5)
with col_end:
    end_z1 = st.slider("End z₁:", -3.0, 3.0, 2.0, 0.5)
    end_z2 = st.slider("End z₂:", -3.0, 3.0, 1.5, 0.5)

# Linear Interpolation
alphas = np.linspace(0.0, 1.0, steps)
interp_points = [
    [start_z1 * (1 - a) + end_z1 * a, start_z2 * (1 - a) + end_z2 * a]
    for a in alphas
]
interp_tensor = torch.tensor(interp_points, dtype=torch.float32)

with torch.no_grad():
    interp_generated = model.decode(interp_tensor)

interp_cols = st.columns(steps)
for i in range(steps):
    img_arr = (interp_generated[i].squeeze().numpy() * 255.0).astype(np.uint8)
    p_img = Image.fromarray(img_arr, mode="L").resize((110, 110), Image.Resampling.NEAREST)
    with interp_cols[i]:
        st.image(p_img, caption=f"Step {i+1}\nz=({interp_points[i][0]:.1f}, {interp_points[i][1]:.1f})", use_container_width=True)
