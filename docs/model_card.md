# Model Card: Text & Image Latent-Space Explorer
**Project P6 — Roll Number: 06**  
**Institution:** MIT-ADT University  
**Date:** Academic Year 2026–27  

---

## 1. Model Details

### 1.1 Model Names & Architectures
1. **Sentence-BERT (`sentence-transformers/all-MiniLM-L6-v2`)**
   - **Type:** Bidirectional Transformer Sentence Encoder (6 layers, 384 hidden dimensions, 12 attention heads).
   - **Pretrained Corpus:** 1+ billion sentence pairs fine-tuned on contrastive semantic textual similarity (STS).
   - **Output:** Normalized 384-dimensional dense vector $\mathbf{v} \in \mathbb{R}^{384}$.
2. **OpenAI CLIP (`openai/clip-vit-base-patch32`)**
   - **Type:** Dual-Encoder Multimodal Vision-Language Model.
   - **Vision Backbone:** Vision Transformer (ViT-B/32) with 32x32 patch size.
   - **Text Backbone:** 12-layer causal Transformer text encoder.
   - **Output:** Aligned 512-dimensional joint projection space $\mathbf{z} \in \mathbb{R}^{512}$.
3. **Fashion-MNIST Convolutional VAE (Trained in this project)**
   - **Type:** Convolutional Variational Autoencoder with 2-Dimensional Latent Space.
   - **Encoder:** 2 Conv2D layers (32 & 64 filters) + BatchNorm + LeakyReLU + Linear(3136, 128) + 2 Linear heads for $\boldsymbol{\mu}, \log \boldsymbol{\sigma}^2$.
   - **Decoder:** Linear(2, 128) + Linear(128, 3136) + 2 ConvTranspose2D layers (64 & 32 filters) + Sigmoid.
   - **Latent Dimension:** $d = 2$ ($z_1, z_2$).

---

## 2. Intended Use & Scope

### 2.1 Primary Intended Uses
- Educational and research exploration of machine learning latent spaces, embeddings, and generative topologies.
- Visual inspection of high-dimensional manifold reductions via PCA and UMAP.
- Real-time synthesis of Fashion-MNIST garments from continuous 2D coordinates.
- Academic demonstration of multimodal contrastive alignment vs unimodal semantic spaces.
- Empirical quantification of gender-occupation association differentials in pretrained language models.

### 2.2 Out-of-Scope & Prohibited Uses
- Inferring, predicting, or attributing personal characteristics (such as gender, sexual orientation, job capability, or character) to any real person.
- Automated employment candidate screening, credit scoring, or biometric identification.
- Deployment in high-stakes automated decision systems without human oversight and bias mitigation.

---

## 3. Training & Evaluation Datasets

### 3.1 Fashion-MNIST Dataset
- **Authors:** Han Xiao, Kashif Rasul, Roland Vollgraf (Zalando Research, 2017).
- **Size:** 60,000 training images, 10,000 testing images ($28 \times 28$ grayscale pixels).
- **Classes:** 10 fashion product categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
- **License:** MIT License.

### 3.2 Evaluation Metrics
- **Evidence Lower Bound (ELBO) Loss:** Total loss combining pixel-level reconstruction and latent prior divergence.
- **Binary Cross-Entropy (BCE):** Measures pixel-wise reconstruction discrepancy.
- **Kullback-Leibler Divergence ($D_{KL}$):** Measures deviation of approximate posterior $q_\phi(z \mid x)$ from standard normal prior $\mathcal{N}(0, I)$.
- **Silhouette Coefficient:** Evaluates cluster cohesion and separation quality in K-Means.
- **Cosine Association Differential ($\Delta$):** Quantifies relative semantic affinity to gendered anchor probes.

---

## 4. Empirical Performance & Results

- **VAE Test Set Performance (10,000 images):**
  - Mean ELBO Loss: `275.29`
  - Mean BCE Reconstruction Loss: `269.69`
  - Mean KL Divergence: `5.61`
- **Clustering Silhouette Score:** `+0.32` to `+0.48` on domain text corpora.
- **Zero-Shot CLIP Cross-Modal Retrieval:** Accurately identifies sneakers, handbags, coats, and nature imagery from natural language prompts.

---

## 5. Bias, Fairness & Ethical Considerations

- Pretrained language embeddings capture demographic and cultural skews present in web-scale corpora.
- In our controlled bias probe, occupations like *Registered Nurse* and *Fashion Designer* exhibited positive differentials ($\Delta > +0.03$), reflecting historical textual association with female pronouns, whereas *Corporate Executive* exhibited negative differentials ($\Delta < -0.01$).
- **Ethical Directive:** These numerical differentials reflect artifactual training corpus co-occurrences, **not** genuine competencies or normative realities.

---

## 6. Model Licences & Attributions

- **Fashion-MNIST:** MIT License.
- **Sentence-Transformers:** Apache 2.0.
- **CLIP:** MIT License.
- **Project P6 Code:** MIT License.
