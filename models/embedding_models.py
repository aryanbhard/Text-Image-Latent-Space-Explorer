"""
Embedding model wrappers for Sentence-BERT and CLIP.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Union, Tuple
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from utils.config import SBERT_MODEL_NAME, CLIP_MODEL_NAME

# Global singletons for memory efficiency
_SBERT_MODEL = None
_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def get_sbert_model() -> SentenceTransformer:
    """Retrieve or load cached Sentence-BERT model."""
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        _SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME, device="cpu")
    return _SBERT_MODEL


def encode_sentences(texts: List[str], normalize: bool = True) -> np.ndarray:
    """
    Generate real sentence embeddings using Sentence-BERT.
    Returns:
        np.ndarray of shape (N, 384)
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = get_sbert_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


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
    Generate real CLIP text embeddings.
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
    Generate real CLIP image embeddings from PIL images.
    Returns:
        np.ndarray of shape (N, 512)
    """
    if not images:
        return np.empty((0, 512), dtype=np.float32)
    model, processor = get_clip_components()
    # Ensure all images are RGB
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
    # Cosine similarity is dot product for normalized vectors
    return np.matmul(text_embeddings, image_embeddings.T)
