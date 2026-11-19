# Project P6: Text & Image Latent-Space Explorer
### Roll Number: 06 | Generative AI Mini-Project
**University:** MIT-ADT University | **Academic Year:** 2026–27  
**Framework:** Python 3.11 + PyTorch + Streamlit + Hugging Face Transformers

---

## 1. Problem Statement & Overview

This project is officially allocated under **Project P6 (Roll Number 06)** from the curriculum *"Generative AI – 60 LLM Mini-Project Problem Statements"*.

The core objective is to build an end-to-end, scientifically grounded machine learning application that investigates how machines represent, organize, and synthesize text and images across high-dimensional embedding spaces and lower-dimensional continuous latent manifolds.

The system incorporates:
- **Dense Text Representations:** 384-dimensional Sentence-BERT (`all-MiniLM-L6-v2`) embeddings.
- **Multimodal Visual Alignment:** 512-dimensional OpenAI CLIP (`clip-vit-base-patch32`) dual-encoder representations.
- **Dimensionality Reduction:** 2D interactive projection manifolds via **PCA** and **UMAP**.
- **Generative Modelling:** A custom **PyTorch Variational Autoencoder (VAE)** with a 2-dimensional latent space trained on **Fashion-MNIST** for on-the-fly image synthesis.
- **LLM Semantic Clustering:** K-Means clustering interpreted and titled via **Google Gemini LLM** (with deterministic semantic fallback).
- **Responsible AI Probing:** Controlled occupation-gender association bias experiments and ethical analysis.

---

## 2. Five-Unit Syllabus Compliance (U1–U5)

| Unit | Official Requirement | Project Implementation | Primary Page |
| :--- | :--- | :--- | :--- |
| **U1** | Foundations of Generative AI | Sentence-BERT & CLIP embeddings; PCA & UMAP 2D projections; interactive Plotly scatter plots with nearest-neighbor search; Generative vs Discriminative AI comparison; zero-shot vs few-shot prompt comparison. | `pages/1_Text_Explorer.py`<br>`pages/7_GPT_BERT_Lab.py` |
| **U2** | GANs and LLMs | Automated cluster naming and technical explanations via Gemini LLM (with fallback); K-Means clustering; PyTorch Fashion-MNIST VAE training; Transformer multi-head self-attention visualization; Prompt-based learning vs Fine-tuning. | `pages/4_Cluster_Analysis.py`<br>`pages/5_VAE_Latent_Space.py`<br>`pages/7_GPT_BERT_Lab.py` |
| **U3** | GPT, BERT & Multimodal Models | Sentence-BERT text embeddings; CLIP image & text embeddings; combined multimodal visualization; image uploads; text-to-image semantic search; architectural comparison of GPT, BERT, Sentence-BERT, and CLIP. | `pages/2_Image_Explorer.py`<br>`pages/3_Combined_Explorer.py`<br>`pages/7_GPT_BERT_Lab.py` |
| **U4** | Generative AI Tools & Techniques | Real PyTorch VAE implementation on Fashion-MNIST; continuous 2D latent space coordinate decoding; interactive Generation Studio with coordinate sliders; stochastic Gaussian sampling; PNG image downloads. | `pages/5_VAE_Latent_Space.py`<br>`pages/6_Image_Generation.py` |
| **U5** | Responsible AI | Occupation–gender association bias probe; cosine similarity association differentials; bias table and differential chart; limitations documentation; safety notices; model and dataset licensing; Model Card. | `pages/8_Responsible_AI.py`<br>`pages/9_Evaluation.py`<br>`docs/model_card.md` |

---

## 3. Technology Stack

- **Deep Learning & ML:** PyTorch 2.x, Torchvision, Scikit-Learn
- **Foundation Models:**
  - `sentence-transformers/all-MiniLM-L6-v2` (384-D dense text vectors)
  - `openai/clip-vit-base-patch32` (512-D multimodal image & text vectors)
  - Custom Convolutional VAE with 2D Gaussian latent space
- **Dimensionality Reduction:** Scikit-Learn PCA, PyTorch UMAP Engine
- **LLM Integration:** Google Gemini API (`gemini-2.5-flash` / `gemini-1.5-flash`) via `google-genai` & `google-generativeai` with automated semantic fallback
- **Frontend & Visualizations:** Streamlit 1.x, Plotly Graph Objects, Pillow (PIL)
- **Quality Assurance:** Pytest automated test suite (11/11 tests passing)

---

## 4. Project Architecture

```
text-image-latent-space-explorer/
│
├── app.py                      # Main Entrypoint & Home Dashboard
├── requirements.txt            # Pinned Python package dependencies
├── packages.txt                # System packages for cloud deployment
├── README.md                   # Complete project documentation
├── LICENSE                     # MIT Open Source License
├── .env.example                # Example environment variable template
├── .gitignore                  # Git ignore rules for caches and artifacts
│
├── pages/                      # Multi-Page Streamlit Application
│   ├── 1_Text_Explorer.py      # Dense text embedding & nearest neighbors
│   ├── 2_Image_Explorer.py     # CLIP visual representations & similar images
│   ├── 3_Combined_Explorer.py   # Multimodal alignment & cross-modal retrieval
│   ├── 4_Cluster_Analysis.py   # K-Means clustering & LLM cluster naming
│   ├── 5_VAE_Latent_Space.py   # 2D VAE manifold & coordinate decoding
│   ├── 6_Image_Generation.py   # VAE Image synthesis studio & PNG download
│   ├── 7_GPT_BERT_Lab.py       # Self-attention, GPT/BERT comparison, prompting
│   ├── 8_Responsible_AI.py     # Occupation-gender bias probes & safety notices
│   └── 9_Evaluation.py         # Empirical metrics, failure cases & viva prep
│
├── models/                     # Deep Learning Models
│   ├── __init__.py
│   ├── vae.py                  # PyTorch ConvVAE (Encoder, Decoder, Loss)
│   ├── embedding_models.py     # Sentence-BERT & CLIP singletons
│   └── clustering.py           # K-Means, silhouette scores, centroids
│
├── services/                   # Business Logic & Infrastructure Services
│   ├── __init__.py
│   ├── embedding_service.py    # PCA, PyTorch UMAP engine, nearest neighbors
│   ├── image_service.py        # Upload validation, format checks, demo gallery
│   ├── llm_service.py          # Gemini API wrapper, sanitization, fallback
│   └── evaluation_service.py   # Metrics aggregation & system summary
│
├── utils/                      # Helper Utilities & Constants
│   ├── __init__.py
│   ├── config.py               # Paths, model names, sample text & occupations
│   ├── visualization.py        # Plotly 2D scatters, loss curves, bar charts
│   └── bias_analysis.py        # Cosine distance differential calculations
│
├── scripts/                    # Standalone Executable Scripts
│   ├── train_vae.py            # Fashion-MNIST VAE training script
│   ├── evaluate_vae.py         # VAE test set evaluation & metric exporter
│   └── prepare_demo_data.py    # Curated demo image generator
│
├── tests/                      # Automated Verification Test Suite
│   ├── test_embeddings.py      # SBERT & CLIP dimension and normalization tests
│   ├── test_vae.py             # Forward pass, loss calculation, decoding tests
│   ├── test_bias.py            # Bias probe differential validation tests
│   └── test_app.py             # Input sanitization & fallback tests
│
├── artifacts/                  # Persisted Model Weights & JSON Metrics
│   ├── vae_fashion_mnist.pth   # Trained PyTorch VAE weights
│   ├── vae_training_history.json
│   ├── vae_evaluation_metrics.json
│   └── vae_sample_latents.pt
│
├── data/                       # Datasets & Cached Assets
│   ├── FashionMNIST/           # Fashion-MNIST raw data (torchvision)
│   └── sample_images/          # Curated demo images for visual exploration
│
└── docs/                       # Project Documentation & Academic Deliverables
    ├── model_card.md           # Model Card according to Mitchell et al.
    ├── architecture.md         # Detailed architectural & data flow diagram
    ├── evaluation_report.md    # Empirical metrics & failure case analysis
    └── project_report.md       # University submission project report
```

---

## 5. Installation & Setup Instructions

### Prerequisites
- Python 3.10 or 3.11
- Pip package manager
- Windows, macOS, or Linux (CPU or GPU supported)

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/text-image-latent-space-explorer.git
cd text-image-latent-space-explorer
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables (Optional for LLM features)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
*(Note: If no API key is provided, the application automatically uses a transparent, clearly-labelled rule-based semantic analyzer).*

### Step 4: Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 6. Training & Evaluation Scripts

### Train the Fashion-MNIST VAE
To retrain the PyTorch VAE on Fashion-MNIST:
```bash
python scripts/train_vae.py
```
This trains the convolutional VAE for 5 epochs on CPU, computes ELBO loss, and saves `artifacts/vae_fashion_mnist.pth` and `artifacts/vae_training_history.json`.

### Evaluate the VAE
To run test set evaluation on 10,000 Fashion-MNIST test images:
```bash
python scripts/evaluate_vae.py
```

### Execute Automated Test Suite
Run all unit tests using pytest:
```bash
python -m pytest tests/
```
All 11 unit tests pass verifying model shapes, normalization, loss formulations, and bias metrics.

---

## 7. Empirical Results & Documented Failure Cases

### Key Measured Results
1. **Sentence-BERT Vectors (384-D):** Normalized L2-norm = 1.000. Within-domain cosine similarity exceeds `0.65`; cross-domain similarity averages `< 0.20`.
2. **Fashion-MNIST VAE (2D Latent Space):**
   - Test Samples: 10,000 images
   - Mean ELBO Loss: `275.29`
   - Mean BCE Reconstruction Loss: `269.69`
   - Mean KL Divergence: `5.61`
3. **Responsible AI Bias Differential ($\Delta$):**
   - Highest female association: `+0.0837` (*Registered Nurse*)
   - Highest male association: `-0.0152` (*Corporate Executive*)

### Documented Failure Cases & Limitations
- **Failure Case 1 (VAE Reconstruction Blur):** Because 784 pixels are compressed into just 2 scalar numbers ($z_1, z_2$) and optimized via pixel-wise BCE loss, reconstructed garments lose sharp edges and fine texture lines. *Mitigation:* Increase latent dimension ($d \ge 16$) and adopt Perceptual Loss (LPIPS) or a VQ-VAE.
- **Failure Case 2 (Disparate Multimodal Spaces):** Directly calculating cosine similarity between raw Sentence-BERT (384-D) and CLIP image vectors (512-D) produces meaningless distances due to independent metric spaces. *Solution:* Use CLIP's unified dual contrastive space for cross-modal tasks.

---

## 8. Deployment on Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Select **Streamlit** SDK.
3. Push this repository to the Space repository.
4. In Space Settings -> **Variables and Secrets**, add `GEMINI_API_KEY` as a secret.
5. The application boots automatically on CPU instances using the cached PyTorch weights.

---

## 9. License & Attribution

- **Fashion-MNIST:** MIT License (Zalando Research)
- **Sentence-Transformers:** Apache 2.0 License (UKP Lab)
- **OpenAI CLIP:** MIT License (OpenAI)
- **Project P6 Implementation:** MIT License
