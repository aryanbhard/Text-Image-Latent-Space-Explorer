"""
Embedding model wrappers for Sentence-BERT and CLIP.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import sys
try:
    import regex
except ImportError:
    import re
    sys.modules["regex"] = re

import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from utils.config import SBERT_MODEL_NAME, CLIP_MODEL_NAME

# Global singletons for memory efficiency
_SBERT_MODEL = None
_SBERT_TOKENIZER = None
_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def get_sbert_model():
    """Retrieve or load cached Sentence-BERT model and tokenizer."""
    global _SBERT_MODEL, _SBERT_TOKENIZER
    if _SBERT_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME, device="cpu")
        except Exception:
            # Fallback to direct PyTorch Transformers with mean pooling
            _SBERT_TOKENIZER = AutoTokenizer.from_pretrained(SBERT_MODEL_NAME)
            _SBERT_MODEL = AutoModel.from_pretrained(SBERT_MODEL_NAME)
            _SBERT_MODEL.eval()
    return _SBERT_MODEL


def encode_sentences(texts: List[str], normalize: bool = True) -> np.ndarray:
    """
    Generate real sentence embeddings using Sentence-BERT (384-D).
    Returns:
        np.ndarray of shape (N, 384)
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = get_sbert_model()

    # Check if loaded via sentence_transformers
    if hasattr(model, "encode"):
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    # Otherwise execute direct PyTorch mean-pooling (exact mathematical equivalent)
    global _SBERT_TOKENIZER
    if _SBERT_TOKENIZER is None:
        _SBERT_TOKENIZER = AutoTokenizer.from_pretrained(SBERT_MODEL_NAME)

    inputs = _SBERT_TOKENIZER(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        tok_emb = out[0]
        mask = inputs["attention_mask"].unsqueeze(-1).expand(tok_emb.size()).float()
        sum_emb = torch.sum(tok_emb * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = sum_emb / sum_mask
        if normalize:
            mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return mean_pooled.cpu().numpy().astype(np.float32)


def get_clip_components() -> Tuple[CLIPModel, CLIPProcessor]:
    """Retrieve or load cached CLIP model and processor."""
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
        _CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        _CLIP_MODEL.eval()
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return _CLIP_MODEL, _CLIP_PROCESSOR


def encode_clip_texts(texts: List[str], normalize: bool = True) -> np.ndarray:
    """
    Generate real CLIP text embeddings (512-D).
    Returns:
        np.ndarray of shape (N, 512)
    """
    if not texts:
        return np.empty((0, 512), dtype=np.float32)
    model, processor = get_clip_components()
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out = model.text_model(**inputs)
        pool = getattr(out, "pooler_output", None)
        if pool is None:
            pool = out[1] if isinstance(out, tuple) else out
        text_features = model.text_projection(pool)
        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype(np.float32)


def encode_clip_images(images: List[Image.Image], normalize: bool = True) -> np.ndarray:
    """
    Generate real CLIP image embeddings from PIL images (512-D).
    Returns:
        np.ndarray of shape (N, 512)
    """
    if not images:
        return np.empty((0, 512), dtype=np.float32)
    model, processor = get_clip_components()
    rgb_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
    inputs = processor(images=rgb_images, return_tensors="pt")
    with torch.no_grad():
        out = model.vision_model(**inputs)
        pool = getattr(out, "pooler_output", None)
        if pool is None:
            pool = out[1] if isinstance(out, tuple) else out
        image_features = model.visual_projection(pool)
        if normalize:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().astype(np.float32)


def compute_cross_modal_similarity(
    text_embeddings: np.ndarray, image_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise cosine similarities between CLIP text embeddings and CLIP image embeddings.
    Assumes embeddings are normalized.
    Returns:
        similarity matrix of shape (num_texts, num_images)
    """
    if text_embeddings.size == 0 or image_embeddings.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.matmul(text_embeddings, image_embeddings.T)
