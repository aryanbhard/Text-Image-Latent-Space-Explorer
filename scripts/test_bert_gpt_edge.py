"""
Edge-case tests for BERT fill-mask and GPT generation features.
Audit verification script — do not modify project files.
"""
import sys
try:
    import regex
except ImportError:
    import re
    sys.modules["regex"] = re

from transformers import pipeline
import sys

fill = pipeline("fill-mask", model="distilbert-base-uncased")
gen = pipeline("text-generation", model="distilgpt2", pad_token_id=50256)

# --- BERT fill-mask ---
print("=== BERT FILL-MASK EDGE CASES ===")

# Standard
p1 = fill("The capital of France is [MASK].", top_k=5)
scores1 = [p["score"] for p in p1]
print(f"Test 1 (standard):  top token={p1[0]['token_str'].strip()!r}, score={scores1[0]:.4f}")
assert all(0 < s < 1 for s in scores1), "Score out of [0,1]"
assert sum(scores1) < 1.0, "Top-5 should NOT sum to 1 (they are subset of vocab softmax)"
print(f"  Top-5 sum={sum(scores1):.4f} (correct: <1 since these are vocab-subset scores)")

# Verb prediction
p2 = fill("The dog [MASK] the ball.", top_k=3)
print(f"Test 2 (verb pred): top token={p2[0]['token_str'].strip()!r}, score={p2[0]['score']:.4f}")
assert p2[0]["score"] > 0, "Zero score unexpected"

# Science context
p3 = fill("The [MASK] is the powerhouse of the cell.", top_k=3)
print(f"Test 3 (biology):   top token={p3[0]['token_str'].strip()!r}, score={p3[0]['score']:.4f}")
tokens3 = [p["token_str"].strip().lower() for p in p3]
print(f"  Top-3 tokens: {tokens3}")

# Counts match top_k
p4 = fill("She is a great [MASK].", top_k=10)
assert len(p4) == 10, f"Expected 10 predictions, got {len(p4)}"
print(f"Test 4 (top_k=10):  received {len(p4)} predictions OK")

print("ALL BERT TESTS PASSED\n")

# --- GPT generation ---
print("=== DISTILGPT2 GENERATION EDGE CASES ===")

# Normal generation
o1 = gen("Machine learning", max_new_tokens=30, do_sample=True, temperature=0.9, top_p=0.92)
t1 = o1[0]["generated_text"]
print(f"Test 1 (normal):    {t1[:80]!r}...")
assert len(t1) > 15, "Generated text too short"
assert t1.startswith("Machine learning"), "Output must include prompt"

# Deterministic (temperature=0 not supported with do_sample=True, use do_sample=False)
o2 = gen("Neural networks", max_new_tokens=20, do_sample=False)
o3 = gen("Neural networks", max_new_tokens=20, do_sample=False)
assert o2[0]["generated_text"] == o3[0]["generated_text"], "Greedy decode should be deterministic"
print(f"Test 2 (greedy det): {o2[0]['generated_text'][:60]!r}")
print("  Greedy decode is deterministic: PASS")

# Max tokens limit
o4 = gen("Test", max_new_tokens=10, do_sample=False)
tokens_added = len(o4[0]["generated_text"].split()) - 1
print(f"Test 3 (max_new_tokens=10): total words={len(o4[0]['generated_text'].split())}")
assert len(o4[0]["generated_text"].split()) <= 15, "max_new_tokens not respected"
print("  Token limit respected: PASS")

print("ALL GPT TESTS PASSED")
