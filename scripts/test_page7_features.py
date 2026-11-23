"""
Test script for Page 7 features:
- Real attention extraction (all-MiniLM-L6-v2)
- BERT fill-mask (distilbert-base-uncased)
- GPT text generation (distilgpt2)
"""
import sys
try:
    import regex
except ImportError:
    import re
    sys.modules["regex"] = re

from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np

print("=== TEST 1: REAL ATTENTION (all-MiniLM-L6-v2) ===")
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
mdl = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", output_attentions=True)
mdl.eval()
inputs = tok("The bank manager approved the loan.", return_tensors="pt", truncation=True, max_length=24)
with torch.no_grad():
    out = mdl(**inputs)
head0 = out.attentions[-1][0, 0].numpy()
tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
print(f"  Tokens: {tokens}")
print(f"  Attn matrix shape: {head0.shape}")
print(f"  Max weight: {head0.max():.4f} | Row-0 sum: {head0[0].sum():.4f}")
assert abs(head0[0].sum() - 1.0) < 1e-4, "Attention rows should sum to 1"
print("  PASS\n")

print("=== TEST 2: BERT FILL-MASK (distilbert-base-uncased) ===")
fill = pipeline("fill-mask", model="distilbert-base-uncased")
preds = fill("The capital of France is [MASK].", top_k=5)
for p in preds:
    token = p["token_str"].strip()
    score = p["score"]
    seq = p["sequence"]
    print(f"  {token:15s}  score={score:.4f}  -> {seq}")
assert len(preds) == 5, "Expected 5 predictions"
print("  PASS\n")

print("=== TEST 3: GPT TEXT GENERATION (distilgpt2) ===")
gen = pipeline("text-generation", model="distilgpt2", pad_token_id=50256)
out = gen(
    "Generative artificial intelligence is transforming",
    max_new_tokens=50,
    do_sample=True,
    temperature=0.9,
    top_p=0.92,
    num_return_sequences=1,
)
generated = out[0]["generated_text"]
print(f"  Generated text:\n  {generated}")
assert len(generated) > 50, "Generated text too short"
print("  PASS\n")

print("ALL THREE FEATURE TESTS PASSED")
