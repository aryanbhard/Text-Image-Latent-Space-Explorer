# System Architecture: Text & Image Latent-Space Explorer
**Project P6 — Roll Number: 06**  
**MIT-ADT University | Academic Year 2026–27**

---

## 1. High-Level Architectural Diagram

```
+-------------------------------------------------------------------------------------------------------------------+
|                                                 STREAMLIT FRONTEND                                                |
|                                                                                                                   |
|  [Home Dashboard]   [Text Explorer]   [Image Explorer]   [Combined Explorer]   [Cluster Analysis with LLM]        |
|  [VAE Latent Space] [Image Studio]    [GPT/BERT Lab]     [Responsible AI]      [Evaluation Dashboard]             |
+-------------------------------------------------------------------------------------------------------------------+
                                                          │
                                                          ▼
+-------------------------------------------------------------------------------------------------------------------+
|                                                  SERVICE LAYER                                                    |
|                                                                                                                   |
|  • EmbeddingService:      PCA (SVD), PyTorch UMAP Engine (Fuzzy Simplicial Sets), Cosine NearestNeighbors         |
|  • ImageService:          Image Upload Security Validation, RGB Conversion, Curated Demo Gallery                  |
|  • LLMService:            Gemini API (2.5-flash / 1.5-flash), Prompt Sanitization, Semantic Rule-Based Fallback    |
|  • EvaluationService:     System-Wide Metric Aggregation, VAE History Loader, Measurable Findings & Failure Logs  |
+-------------------------------------------------------------------------------------------------------------------+
                                                          │
                                                          ▼
+-------------------------------------------------------------------------------------------------------------------+
|                                                   MODEL LAYER                                                     |
|                                                                                                                   |
|  ┌─────────────────────────┐      ┌─────────────────────────┐      ┌───────────────────────────────────────────┐  |
|  │  SentenceTransformer    │      │       CLIP Dual Model   │      │       PyTorch FashionVAE (Trained)        │  |
|  │  • all-MiniLM-L6-v2     │      │  • ViT-B/32 Vision Enc. │      │  • Encoder: Conv2D -> (32, 64) -> FC128   │  |
|  │  • 384-D Dense Vectors  │      │  • Transformer Text Enc.│      │  • Latent: mu, logvar in R^2              │  |
|  │  • Normalization (L2)   │      │  • 512-D Aligned Space  │      │  • Decoder: FC128 -> ConvTranspose2D -> X │  |
|  └─────────────────────────┘      └─────────────────────────┘      └───────────────────────────────────────────┘  |
+-------------------------------------------------------------------------------------------------------------------+
                                                          │
                                                          ▼
+-------------------------------------------------------------------------------------------------------------------+
|                                           DATA & PERSISTENCE LAYER                                                |
|                                                                                                                   |
|  • artifacts/vae_fashion_mnist.pth          • artifacts/vae_training_history.json                                 |
|  • artifacts/vae_evaluation_metrics.json    • artifacts/vae_sample_latents.pt                                     |
|  • data/FashionMNIST/ (Raw test/train)      • data/sample_images/ (Curated 12-image visual gallery)               |
+-------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Data Flow & Processing Pipelines

### 2.1 Text Embedding Pipeline
1. **Raw Text Input:** User enters sentences or selects the curated 25-sentence corpus across 5 domains.
2. **Tokenization:** Subword WordPiece tokenization via Hugging Face tokenizers.
3. **Transformer Encoding:** 6-layer MiniLM encoder generates contextual token representations.
4. **Mean Pooling & Normalization:** Mean pooling over token embeddings produces a single 384-dimensional vector with unit length $\|\mathbf{v}\| = 1.0$.
5. **Projection:** PCA computes principal axes of maximum variance; PyTorch UMAP minimizes fuzzy cross-entropy to create 2D coordinates.
6. **Neighborhood Query:** `NearestNeighbors` identifies closest semantic vectors via cosine distance.

### 2.2 Multimodal Contrastive Alignment Pipeline
1. **Image Input:** 224x224 RGB image tensors normalized by ImageNet mean and std.
2. **Text Input:** Natural language prompt tokenized with causal BPE.
3. **Dual Forward Pass:** ViT-B/32 processes image patches; Transformer processes text tokens.
4. **Linear Projections:** `text_projection` and `visual_projection` map both outputs into $\mathbb{R}^{512}$.
5. **Cross-Modal Dot Product:** Normalized dot product $\mathbf{t} \cdot \mathbf{i}^T$ yields cosine similarity matrix.

### 2.3 Variational Autoencoder (VAE) Generative Pipeline
1. **Input:** Grayscale Fashion-MNIST image $x \in [0, 1]^{1 \times 28 \times 28}$.
2. **Inference (Encoder):** Conv2D layers compress $x$ to feature map $(64, 7, 7)$, followed by fully-connected projection to latent parameters $\boldsymbol{\mu} \in \mathbb{R}^2$ and $\log \boldsymbol{\sigma}^2 \in \mathbb{R}^2$.
3. **Stochastic Sampling (Reparameterization Trick):**
   $$z = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
4. **Synthesis (Decoder):** Dense layers expand $z \in \mathbb{R}^2$ back to $(64, 7, 7)$, followed by two ConvTranspose2D layers and Sigmoid activation, yielding reconstructed image $\hat{x} \in [0, 1]^{1 \times 28 \times 28}$.
5. **Loss Backpropagation:**
   $$\mathcal{L}_{\text{ELBO}} = \text{BCE}(x, \hat{x}) + D_{KL}(q(z \mid x) \parallel \mathcal{N}(0, I))$$
