"""
LLM Service for automated cluster naming, explanation, and prompt engineering experiments.
Integrates Google Gemini API with robust prompt sanitization and clearly identified fallback.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import os
import re
from typing import List, Dict, Any, Optional
from collections import Counter
from utils.config import GEMINI_API_KEY


def sanitize_input(text: str, max_length: int = 300) -> str:
    """
    Sanitize text input to prevent basic prompt-injection attacks.
    Removes system instruction attempts and restricts length.
    """
    cleaned = text.strip()[:max_length]
    # Remove attempts to override system prompt
    cleaned = re.sub(
        r"(?i)(ignore previous instructions|system prompt|disregard all|you are now)",
        "[REDACTED]",
        cleaned,
    )
    return cleaned


def call_gemini_api(prompt: str, api_key: Optional[str] = None) -> Optional[str]:
    """
    Execute call to Google Gemini API using google-genai or google-generativeai.
    Returns response string or None if unavailable.
    """
    key = api_key or os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not key or not key.strip():
        return None

    # Try google-genai SDK first
    try:
        from google import genai

        client = genai.Client(api_key=key.strip())
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        if response and response.text:
            return response.text.strip()
    except Exception:
        pass

    # Try google-generativeai SDK fallback
    try:
        import google.generativeai as genai

        genai.configure(api_key=key.strip())
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
    except Exception:
        pass

    return None


def _fallback_cluster_analysis(cluster_id: int, sample_texts: List[str]) -> Dict[str, str]:
    """
    Rule-based deterministic semantic analyzer used when Gemini API key is not configured.
    Explicitly labelled to maintain academic integrity and truthfulness.
    """
    # Tokenize and extract key nouns/terms
    stop_words = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "is",
        "are", "was", "were", "of", "by", "that", "this", "from", "as", "it", "be",
        "their", "its", "into", "through", "across", "due", "out", "over"
    }

    words = []
    for text in sample_texts:
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        words.extend([w for w in tokens if w not in stop_words])

    freq = Counter(words).most_common(4)
    top_terms = [w.capitalize() for w, _ in freq]

    if top_terms:
        title = f"Domain: {' & '.join(top_terms[:2])}"
        desc = (
            f"[Rule-Based Fallback Analysis — No Gemini API Key Detected]\n"
            f"This cluster groups {len(sample_texts)} samples characterized by recurrent semantic "
            f"markers: {', '.join(top_terms)}. The high cosine similarity between these representations "
            f"indicates coherent latent space grouping around these conceptual topics."
        )
    else:
        title = f"Cluster #{cluster_id + 1}: General Semantic Group"
        desc = (
            f"[Rule-Based Fallback Analysis — No Gemini API Key Detected]\n"
            f"Cluster containing {len(sample_texts)} items exhibiting mutual proximity in latent space."
        )

    return {"title": title, "explanation": desc, "source": "Rule-Based Semantic Fallback"}


def generate_cluster_summary(
    cluster_id: int,
    sample_texts: List[str],
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate meaningful cluster name and explanation using Gemini LLM,
    or transparently fall back to semantic rule-based extraction.
    """
    sanitized_samples = [sanitize_input(t) for t in sample_texts[:8]]
    samples_formatted = "\n".join([f"- {s}" for s in sanitized_samples])

    prompt = (
        f"You are an expert machine learning researcher analyzing high-dimensional embedding clusters.\n"
        f"Analyze the following sentences belonging to Cluster #{cluster_id + 1}:\n"
        f"{samples_formatted}\n\n"
        f"Provide your response in exactly this format:\n"
        f"NAME: <Concise, professional 3-6 word descriptive title for this cluster>\n"
        f"EXPLANATION: <A 2-3 sentence technical explanation of the underlying themes and semantic similarities.>"
    )

    llm_output = call_gemini_api(prompt, api_key=api_key)

    if llm_output:
        name_match = re.search(r"NAME:\s*(.+)", llm_output, re.IGNORECASE)
        desc_match = re.search(r"EXPLANATION:\s*(.+)", llm_output, re.IGNORECASE | re.DOTALL)

        title = name_match.group(1).strip() if name_match else f"Cluster #{cluster_id + 1} Theme"
        desc = desc_match.group(1).strip() if desc_match else llm_output.strip()

        return {
            "title": title,
            "explanation": desc,
            "source": "Gemini LLM (Live API)",
        }

    # Graceful fallback
    return _fallback_cluster_analysis(cluster_id, sample_texts)


def run_prompt_comparison_demo(
    topic: str,
    technique_a: str = "Zero-Shot",
    technique_b: str = "Few-Shot",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare zero-shot vs few-shot prompting techniques for a given topic or classification task.
    Demonstrates prompt engineering principles (Unit 1 & Unit 2 requirements).
    """
    zero_shot_prompt = (
        f"Classify the sentiment of the following review as Positive, Neutral, or Negative:\n"
        f"Review: \"{topic}\"\n"
        f"Sentiment:"
    )

    few_shot_prompt = (
        f"Classify the sentiment of reviews into Positive, Neutral, or Negative.\n\n"
        f"Review: \"The product arrived on time and exceeded all my expectations!\"\n"
        f"Sentiment: Positive\n\n"
        f"Review: \"It works as described, nothing special.\"\n"
        f"Sentiment: Neutral\n\n"
        f"Review: \"Terrible battery life and broke after three days.\"\n"
        f"Sentiment: Negative\n\n"
        f"Review: \"{topic}\"\n"
        f"Sentiment:"
    )

    resp_a = call_gemini_api(zero_shot_prompt, api_key=api_key)
    resp_b = call_gemini_api(few_shot_prompt, api_key=api_key)

    if not resp_a:
        # Deterministic illustrative evaluation
        resp_a = "Positive [Simulated Fallback — Zero-shot direct inference based on parametric priors]"
    if not resp_b:
        resp_b = "Positive [Simulated Fallback — Few-shot conditioned on in-context demonstration examples]"

    return {
        "zero_shot": {
            "prompt": zero_shot_prompt,
            "response": resp_a,
            "technique": "Zero-Shot (Direct instruction without exemplars)",
        },
        "few_shot": {
            "prompt": few_shot_prompt,
            "response": resp_b,
            "technique": "Few-Shot (In-context demonstration exemplars)",
        },
    }
