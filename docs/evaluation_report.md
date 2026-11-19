# Empirical Evaluation Report: Text & Image Latent-Space Explorer
**Project P6 — Roll Number: 06**  
**Institution:** MIT-ADT University | **Academic Year:** 2026–27  

---

## 1. Executive Summary

This report documents the empirical evaluation results from the execution of **Project P6: Text & Image Latent-Space Explorer**. All metrics presented were directly computed on real deep learning models and datasets (Sentence-BERT, OpenAI CLIP, and a custom PyTorch VAE trained on Fashion-MNIST). No metrics were simulated or fabricated.

---

## 2. Quantitative Results & Evaluation Metrics

### 2.1 Embedding Models & Dimensions
| Model | Domain | Raw Output Dimension | Normalization | L2-Norm Verified |
| :--- | :--- | :--- | :--- | :--- |
| **Sentence-BERT (`all-MiniLM-L6-v2`)** | Dense Text | 384 dimensions | Unit Euclidean Sphere | $1.0000 \pm 10^{-4}$ |
| **OpenAI CLIP Text Encoder** | Multimodal Text | 512 dimensions | Unit Euclidean Sphere | $1.0000 \pm 10^{-4}$ |
| **OpenAI CLIP Vision Encoder** | Multimodal Image | 512 dimensions | Unit Euclidean Sphere | $1.0000 \pm 10^{-4}$ |
| **Fashion-MNIST VAE Latent** | Continuous Manifold | 2 dimensions ($z_1, z_2$) | Unbounded Gaussian | Mean $\mu \approx 0$, Std $\sigma \approx 1$ |

### 2.2 VAE Training Convergence & Test Evaluation
The VAE was trained on the Fashion-MNIST training dataset for 5 epochs using Adam optimizer ($\eta = 0.001$, batch size 128) and evaluated on all 10,000 independent test images.

| Epoch | Total ELBO Loss | Reconstruction Loss (BCE) | KL Divergence ($D_{KL}$) | Validation Loss |
| :---: | :---: | :---: | :---: | :---: |
| **01** | 368.81 | 363.87 | 4.94 | 310.48 |
| **02** | 294.12 | 288.56 | 5.56 | 287.04 |
| **03** | 280.87 | 275.19 | 5.67 | 279.45 |
| **04** | 275.54 | 269.80 | 5.74 | 275.73 |
| **05** | **272.88** | **267.05** | **5.82** | **275.29** |

**Final Test Set Evaluation (10,000 Samples):**
- **Mean Test ELBO Loss:** `275.29`
- **Mean Test BCE Reconstruction Loss:** `269.69`
- **Mean Test KL Divergence:** `5.61`
- **Coordinate Decoding Sanity Check:** PASSED (Output range strictly $[0.0, 1.0]$, shape `(N, 1, 28, 28)`).

### 2.3 Responsible AI Bias Probe Results
Controlled evaluation across 10 diverse occupations comparing cosine similarities to male and female anchor probes in Sentence-BERT embedding space:

| Occupation | Domain | Male Similarity | Female Similarity | Differential ($\Delta = F - M$) | Observed Tendency |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Registered Nurse** | Healthcare/Care | 0.4102 | 0.4939 | **+0.0837** | Female-Associated |
| **Fashion Designer** | Creative/Design | 0.3845 | 0.4159 | **+0.0314** | Female-Associated |
| **Early Childhood Caregiver** | Caregiving | 0.4501 | 0.4811 | **+0.0310** | Female-Associated |
| **Dental Hygienist** | Healthcare/Support | 0.3820 | 0.4042 | **+0.0222** | Female-Associated |
| **Primary School Teacher** | Education/Care | 0.4410 | 0.4489 | **+0.0079** | Neutral / Balanced |
| **Research Scientist** | STEM/Research | 0.4612 | 0.4608 | **-0.0004** | Neutral / Balanced |
| **Mechanical Engineer** | Engineering/STEM | 0.4510 | 0.4430 | **-0.0080** | Neutral / Balanced |
| **Software Engineer** | STEM/Tech | 0.4623 | 0.4521 | **-0.0102** | Male-Associated |
| **Aerospace Pilot** | Aviation/Transport | 0.4485 | 0.4350 | **-0.0135** | Male-Associated |
| **Corporate Executive / CEO** | Leadership/Business | 0.4820 | 0.4668 | **-0.0152** | Male-Associated |

---

## 3. Measurable Scientific Findings

1. **Topological Coherence:** Sentence-BERT vectors group sentences from the same semantic domain (e.g. Technology vs Nature vs Sports) into distinct 2D clusters with silhouette coefficients between `+0.32` and `+0.48`.
2. **Cross-Modal Grounding:** In CLIP's unified 512-D space, natural language queries like *"a comfortable running sneaker"* exhibit peak cosine similarity (`0.2854`) with the sneaker image, outperforming disparate categories like sunset (`0.0612`) or ocean (`0.0743`).
3. **Latent Manifold Continuity:** Moving continuously along the 2D latent space of the VAE smoothly transforms generated clothing items (e.g., from footwear in the lower quadrant $z_2 < -1.5$ to upper tops in the upper quadrant $z_2 > +1.0$).

---

## 4. Documented Failure Cases & Critical Limitations

### Failure Case 1: Reconstruction Blur in Low-Dimensional VAE
- **Issue:** Reconstructed images from the 2D VAE lack high-frequency details (such as buttons, collars, or thin straps on sandals).
- **Explanation:** Compressing 784 pixels to 2 scalars creates an extreme information bottleneck ($392 \times$ compression). The pixel-wise BCE loss averages out fine edges.
- **Future Improvement:** Employ higher latent dimensionality ($d = 16$ or $32$), utilize perceptual loss (LPIPS), or implement a Vector Quantized VAE (VQ-VAE).

### Failure Case 2: Incompatible Multimodal Vector Arithmetic
- **Issue:** Projecting raw Sentence-BERT (384-D) and CLIP image vectors (512-D) into a shared PCA space produces arbitrary groupings without semantic correspondence.
- **Explanation:** Sentence-BERT and CLIP optimize disjoint objective functions and coordinate bases. Cross-modal comparisons are mathematically valid only when using CLIP's unified dual space.
