# Demo Script — Text & Image Latent-Space Explorer (P6)
**Student:** Aryan Bhardwaj | **Roll No.:** 06 | **Project:** P6  
**Duration:** 2–3 minutes  
**University:** MIT ADT University, Pune | **Guide:** Dr. Snehalata Funde

---

## PRE-DEMO SETUP

Before the demo session:
1. Run: `python -m streamlit run app.py`
2. Open `http://localhost:8501` in Chrome (full screen, Zoom 90%)
3. Ensure all 9 pages are accessible from the sidebar
4. Check that GEMINI_API_KEY is set in environment (optional but preferred for Cluster Analysis page)

---

## NARRATION SCRIPT

---

### [0:00–0:20] OPENING — Home Page

*[Show the app.py home page]*

> "Good morning, everyone. I am Aryan Bhardwaj, Roll Number 06. My project is the **Text and Image Latent-Space Explorer** — an interactive educational platform that demonstrates five curriculum units of AI and Analytics through genuine, real-time model inference.
>
> The home page shows the five units covered and the mathematical formulas behind each feature. All results in this application come from actual deep learning inference — there is no hardcoded or simulated data."

---

### [0:20–0:50] UNIT 1 — Text Embedding Explorer (Page 1)

*[Navigate to Page 1: Text Explorer]*

> "Unit 1 covers latent spaces and embeddings. Here, I am running **Sentence-BERT** — the `all-MiniLM-L6-v2` model — which produces 384-dimensional vectors for each sentence.
>
> You can see 25 sentences from five domains — Culinary, Space, Engineering, Marine Biology, and Music — projected into 2D using PCA. Watch how the sentences cluster by domain — this is the geometric structure of the learned latent space.
>
> *[Switch to UMAP]* With UMAP, we see even tighter clusters, as UMAP preserves non-linear manifold structure.
>
> *[Select a sentence, show nearest neighbors]* When I select 'The chef prepared a five-course meal,' its nearest neighbors are other culinary sentences — confirming the semantic meaning is encoded geometrically."

---

### [0:50–1:15] UNIT 1 — CLIP + Combined Explorer (Pages 2–3)

*[Navigate to Page 3: Combined Explorer, Mode 1]*

> "CLIP takes this further — it aligns text and images in the **same** 512-dimensional space. Here I can type any text query and retrieve visually similar images. Watch —
>
> *[Type: 'a comfortable running sneaker or sports shoe', click]*
>
> CLIP correctly ranks the sneaker images highest, without any fine-tuning. This is zero-shot cross-modal retrieval — text and images in a shared geometric manifold."

---

### [1:15–1:40] UNIT 2+3 — Transformer Attention & BERT (Page 7)

*[Navigate to Page 7, Tab 2: Real Attention Visualizer]*

> "Unit 2 covers transformer architecture. This tab extracts **real attention weights** from the `all-MiniLM-L6-v2` model's forward pass.
>
> *[Enter a sentence, click 'Extract Real Attention Weights']*
>
> This heatmap shows which tokens each query token attends to — extracted directly from Layer 6, Head 0. The row sums equal 1.0 — these are real softmax attention weights.
>
> *[Switch to Tab 3: BERT Masked Prediction]*
>
> Unit 3 — Language Models. Here, `distilbert-base-uncased` predicts masked tokens using **bidirectional context**. 'The capital of France is [MASK].' — BERT predicts 'paris' with 99.5% confidence."

---

### [1:40–2:05] UNIT 4 — VAE Latent Space (Pages 5–6)

*[Navigate to Page 5: VAE Latent Space]*

> "Unit 4 is the Variational Autoencoder. I trained a custom **FashionVAE** — 844,805 parameters — on Fashion-MNIST. This 2D scatter shows 1,000 test images encoded into their 2-dimensional latent coordinates, colored by class.
>
> *[Adjust z1, z2 sliders]*
>
> When I move these sliders, the decoder generates a new Fashion-MNIST image in real time — directly from the trained PyTorch model. The smooth transitions between garment classes demonstrate the continuous latent manifold learned by the VAE.
>
> The test ELBO is 275.29, with BCE = 269.69 and KLD = 5.61, verified on 10,000 test samples."

---

### [2:05–2:30] UNIT 5 — Responsible AI (Page 8)

*[Navigate to Page 8: Responsible AI]*

> "Unit 5 addresses Responsible AI. This probe computes cosine-similarity differentials between occupation embeddings and gender anchor centroids.
>
> *[Show bias table and chart]*
>
> 'Registered Nurse' has the highest female association — delta of +0.0837. 'Corporate Executive' shows a mild male association. These are real mathematical biases encoded in the pretrained model's training corpus — Wikipedia and Common Crawl.
>
> The application prominently displays ethical notices: these are statistical artifacts, not individual facts, and must never be used to infer individual identity or competence."

---

### [2:30–2:50] CLOSING — Evaluation Dashboard (Page 9)

*[Navigate to Page 9: Evaluation]*

> "Finally, Page 9 shows the complete evaluation dashboard — all metrics verified by running the automated test suite.
>
> *[Show pytest results]*
>
> 11 out of 11 tests pass. This is a fully verified, genuine implementation across all five curriculum units.
>
> Thank you."

---

## BACKUP TALKING POINTS

If asked about specific technical details during Q&A:

**Q: Why UMAP over t-SNE?**  
A: UMAP is O(n log n) vs O(n²) for t-SNE. It preserves global manifold structure, not just local neighborhoods. Our implementation uses pure PyTorch — no C-extension dependencies.

**Q: What is the ELBO and why does it matter?**  
A: ELBO = Reconstruction Loss (BCE) + KL Divergence. It is the tractable lower bound on the evidence (log p(x)). Maximizing ELBO simultaneously minimizes reconstruction error and regularizes the posterior to match the Gaussian prior.

**Q: Why only 2D latent space for the VAE?**  
A: 2D enables interactive human visualization. It demonstrates the core concept — compressing 784 pixel dimensions to 2 — at the cost of reconstruction sharpness. With d=16+, reconstruction quality improves significantly.

**Q: Is the attention extraction real or simulated?**  
A: Real. `AutoModel.from_pretrained(..., output_attentions=True)` stores the actual softmax attention tensors. We read `outputs.attentions[-1][0, head_idx]` from the last encoder layer's real forward pass.

**Q: What happens without a Gemini API key?**  
A: The application uses a transparent rule-based semantic analyzer for cluster naming (clearly labeled). distilgpt2 is used for text generation. All five units work fully without any API key.

---

*Demo Script — Aryan Bhardwaj | P6 | MIT ADT University | November 2026*
