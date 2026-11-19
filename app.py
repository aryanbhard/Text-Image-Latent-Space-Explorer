"""
Text & Image Latent-Space Explorer
Roll Number: 06 | Generative AI Mini-Project (Project P6)
MIT-ADT University
"""

import streamlit as st
from PIL import Image
from utils.config import (
    SBERT_MODEL_NAME,
    CLIP_MODEL_NAME,
    VAE_LATENT_DIM,
    VAE_WEIGHTS_PATH,
    ARTIFACTS_DIR,
)
from services.evaluation_service import get_system_evaluation_summary
from utils.visualization import render_table_as_markdown

st.set_page_config(
    page_title="Text & Image Latent-Space Explorer",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for academic, professional styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 1.1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .syllabus-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #ffffff;
        background-color: #2563eb;
        border-radius: 4px;
        margin-right: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar metadata
st.sidebar.title("🌌 Latent-Space Explorer")
st.sidebar.markdown("**Project P6 — Roll No: 06**")
st.sidebar.markdown("*Generative AI Mini-Project*")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Syllabus Navigation")
st.sidebar.info(
    "Use the pages above to explore text embeddings, image representations, "
    "multimodal alignment, VAE generation, and bias probes."
)
st.sidebar.markdown("---")
st.sidebar.caption("MIT-ADT University • Academic Year 2026–27")

# Header Section
st.markdown('<div class="main-title">TEXT & IMAGE LATENT-SPACE EXPLORER</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Discover how machines understand, organise, and generate text and images through high-dimensional latent representations.</div>',
    unsafe_allow_html=True,
)

# Top Status Indicators
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Text Encoder", "Sentence-BERT", "384-D Vectors")
with col2:
    st.metric("Multimodal Encoder", "OpenAI CLIP", "512-D Vectors")
with col3:
    st.metric("Generative VAE", "Fashion-MNIST", "2-D Latent Space")
with col4:
    vae_ready = VAE_WEIGHTS_PATH.exists()
    st.metric("VAE Status", "Trained & Ready" if vae_ready else "Needs Training", "PyTorch weights")

st.markdown("---")

# Main Dashboard Tabs
tab_overview, tab_syllabus, tab_lifecycle, tab_theory = st.tabs([
    "📋 Project Overview",
    "🎯 Syllabus Coverage (U1–U5)",
    "🔄 Generative AI Lifecycle",
    "🧠 Key Mathematical Concepts",
])

with tab_overview:
    st.subheader("Project Objectives & Architecture")
    st.write(
        """
        This application provides an interactive laboratory to inspect how foundational 
        deep learning models project continuous semantic concepts into geometric coordinates (**latent space**).
        """
    )
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            #### 🌟 Core Capabilities
            1. **Dense Text Embedding Space:** Real-time generation of 384-dimensional Sentence-BERT embeddings.
            2. **Visual & Multimodal Space:** 512-dimensional CLIP image embeddings and joint cross-modal space.
            3. **Dimensionality Reduction:** 2D interactive projections via **PCA** and **UMAP**.
            4. **Cluster Discovery & LLM Naming:** K-Means clustering analyzed and explained via **Gemini LLM**.
            5. **Real PyTorch VAE:** Continuous 2D manifold learned on Fashion-MNIST with interactive image decoding.
            6. **Image Generation Studio:** Coordinate-based and stochastic synthesis of Fashion-MNIST garments.
            7. **Responsible AI & Bias Probes:** Quantifying gender-occupation semantic associations in embedding spaces.
            """
        )
    with col_b:
        st.markdown(
            """
            #### 🛠️ Technology Stack
            - **Framework:** Python 3.11 + Streamlit UI
            - **Deep Learning:** PyTorch 2.x, Torchvision
            - **Language Models:** Hugging Face Transformers, Sentence-Transformers (`all-MiniLM-L6-v2`)
            - **Multimodal:** OpenAI CLIP (`clip-vit-base-patch32`)
            - **Manifold Learning:** Scikit-Learn PCA, K-Means, NearestNeighbors, PyTorch UMAP Engine
            - **LLM Integration:** Google Gemini API (`gemini-2.5-flash`) with transparent rule-based fallback
            - **Visualization:** Plotly Interactive Graph Objects
            """
        )

with tab_syllabus:
    st.subheader("Official University Syllabus Compliance (Units 1–5)")
    st.write("Every required component is implemented with real deep learning models and verifiable calculations:")
    
    syllabus_data = [
        {"Unit": "Unit 1", "Title": "Foundations of Generative AI", "Features": "Sentence-BERT embeddings, PCA, UMAP 2D projections, interactive scatter plots, nearest-neighbour search, generative vs discriminative analysis, zero-shot vs few-shot prompt comparison.", "Page": "1_Text_Explorer & 7_GPT_BERT_Lab"},
        {"Unit": "Unit 2", "Title": "GANs and LLMs", "Features": "K-Means clustering, LLM automated cluster naming and explanation, Fashion-MNIST VAE training, encoder/decoder architecture, transformer attention mechanism, prompt-based learning vs fine-tuning.", "Page": "4_Cluster_Analysis & 5_VAE_Latent_Space & 7_GPT_BERT_Lab"},
        {"Unit": "Unit 3", "Title": "GPT, BERT and Multimodal Models", "Features": "Sentence-BERT vs CLIP text & image embeddings, joint multimodal 2D visualization, image uploads, genuine cross-modal semantic similarity, architectural comparison between GPT, BERT, and CLIP.", "Page": "2_Image_Explorer & 3_Combined_Explorer & 7_GPT_BERT_Lab"},
        {"Unit": "Unit 4", "Title": "Generative AI Tools & Techniques", "Features": "PyTorch VAE implementation, 2D latent space coordinate decoding, interactive generation studio, latent slider controls, random seed sampling, image gallery with PNG download.", "Page": "5_VAE_Latent_Space & 6_Image_Generation"},
        {"Unit": "Unit 5", "Title": "Responsible AI", "Features": "Occupation–gender bias probe on sentence embeddings, cosine similarity association differentials, bias table and visualization, safety notices, limitations, and Model Card documentation.", "Page": "8_Responsible_AI & 9_Evaluation"},
    ]
    st.markdown(render_table_as_markdown(syllabus_data), unsafe_allow_html=True)

with tab_lifecycle:
    st.subheader("The Generative AI Lifecycle")
    st.markdown(
        """
        ```
        +-------------------------------------------------------------------------------------------------------+
        |                                       GENERATIVE AI LIFECYCLE                                         |
        +-------------------------------------------------------------------------------------------------------+
           [1. Data Collection]      Raw text corpora & visual datasets (Fashion-MNIST, Natural Imagery)
                    │
                    ▼
           [2. Preprocessing]        Tokenization, BPE subwords, image resizing (224x224 RGB, 28x28 grayscale)
                    │
                    ▼
           [3. Representation]       High-dimensional embedding projection (Sentence-BERT 384-D, CLIP 512-D)
                    │
                    ▼
           [4. Manifold Learning]    Dimensionality reduction (PCA linear variance, UMAP non-linear topology)
                    │
                    ▼
           [5. Latent Exploration]   Clustering (K-Means), continuous latent sampling (VAE 2D Gaussian manifold)
                    │
                    ▼
           [6. Generation]           Decoding latent coordinates into synthetic imagery via transposed convolutions
                    │
                    ▼
           [7. Evaluation]           Reconstruction loss (BCE), KL divergence, silhouette scores, nearest neighbors
                    │
                    ▼
           [8. Responsible AI]       Bias probing (association differentials), safety notices, ethical model cards
        +-------------------------------------------------------------------------------------------------------+
        ```
        """
    )

with tab_theory:
    st.subheader("Foundational Concepts & Mathematical Formulations")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            r"""
            #### 1. Generative vs Discriminative AI
            - **Discriminative Models:** Model the posterior probability $P(Y \mid X)$. They learn decision boundaries to separate classes (e.g., classifying a garment as a coat or shoe).
            - **Generative Models:** Model the joint probability distribution $P(X, Y)$ or the data distribution $P(X)$. They capture how data is generated and can sample new instances $\tilde{x} \sim P(X)$.
            
            #### 2. Latent Space Representation
            A latent space is a lower-dimensional manifold $\mathcal{Z} \subset \mathbb{R}^d$ where semantically similar data points lie near one another. Rather than operating in raw pixel space ($\mathbb{R}^{784}$ or $\mathbb{R}^{150528}$), models learn compact representations where arithmetic operations correlate with semantic transformations.
            """
        )
    with col_t2:
        st.markdown(
            r"""
            #### 3. Variational Autoencoders (VAE)
            VAEs impose a Gaussian prior $p(z) = \mathcal{N}(0, I)$ on the latent coordinates.
            The model is trained by maximizing the **Evidence Lower Bound (ELBO)**:
            $$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{KL}(q_\phi(z \mid x) \parallel p(z))$$
            
            - **Reconstruction Loss:** Measures pixel fidelity between input $x$ and output $\hat{x}$.
            - **KL Divergence:** Enforces smooth, continuous latent coverage without gaps.
            - **Reparameterization Trick:** Enables gradient backpropagation through stochastic sampling:
              $$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
            """
        )

st.markdown("---")
st.info("👈 **Select a page from the left sidebar** to begin interactive exploration of text embeddings, image spaces, clustering, and the VAE.")
