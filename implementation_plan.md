# Project Implementation Plan

## Phase 1 – Project Audit
- Inspect workspace – empty directory confirmed.
- Identify required components per specification.
- Create folder structure and placeholder files.
- Document risks and model requirements.

## Phase 2 – Core Implementation
- Build Streamlit multi‑page app.
- Integrate Sentence‑BERT (via `sentence-transformers`).
- Integrate CLIP (via `openai/clip-vit-base-patch32`).
- Implement PCA (`sklearn.decomposition.PCA`) and UMAP (`umap-learn`).
- Add interactive Plotly scatter plots with nearest‑neighbour search (`sklearn.neighbors.NearestNeighbors`).
- Cache models on first use.

## Phase 3 – VAE Implementation
- PyTorch VAE with 2‑D latent space trained on Fashion‑MNIST.
- Training script (`scripts/train_vae.py`) with reproducible seed.
- Save model weights in `models/vae.pth`.
- Visualise loss curves and latent space.
- Interactive decoder page.

## Phase 4 – Clustering & LLM
- K‑Means clustering (`sklearn.cluster.KMeans`).
- LLM service using Gemini API (fallback to dummy naming).
- Prompt‑comparison demo.
- GPT/BERT demonstration page.

## Phase 5 – Responsible AI
- Occupation‑gender bias probe using Sentence‑BERT embeddings.
- Cosine similarity scoring, bar chart.
- Safety notice and limitations.

## Phase 6 – Frontend Integration
- Streamlit navigation sidebar linking all pages.
- Consistent styling, cards, tooltips.
- Loading spinners and error handling.

## Phase 7 – Testing
- Unit tests for VAE, embeddings, bias analysis.
- CI‑style script to run `pytest`.
- Verify model downloads and API fallback.

## Phase 8 – Documentation
- README, MODEL_CARD.md, architecture diagram, report markdown.
- Capture real screenshots after running the app.

## Phase 9 – Deployment
- Hugging Face Spaces configuration (`requirements.txt`, `setup.sh` optional).
- Environment variable `GEMINI_API_KEY`.
- Verify CPU‑only execution.

## Phase 10 – Final Audit
- Checklist of all syllabus units.
- List of completed vs pending items.
- Deployment URL and GitHub repo placeholder.

### Technical Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Large transformer models exceed CPU memory | Training/inference may be slow or OOM | Use `sentence-transformers/all-MiniLM-L6-v2` and CLIP `openai/clip-vit-base-patch32` (both CPU‑friendly). |
| Gemini API quota / downtime | Cluster naming fails | Provide deterministic fallback (e.g., “Cluster #i”). |
| VAE training time | Long training may block demo | Pre‑train once, store weights; provide a script for re‑training if needed. |
| File‑upload abuses | Security risk | Validate MIME type, size <5 MB, store temporarily only. |
| Dependency conflicts | Install failures | Pin versions in `requirements.txt`. |

---

*This plan will guide the incremental development. Each phase will be implemented, tested, and committed before moving to the next.*