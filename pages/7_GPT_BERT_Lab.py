"""
PAGE 7: GPT, BERT and Attention Laboratory (Unit 1, Unit 2 & Unit 3 Requirements)
Contrasts autoregressive GPT generation vs bidirectional BERT representation,
demonstrates REAL multi-head self-attention from a pretrained Transformer (all-MiniLM-L6-v2),
adds BERT Masked-Language-Model prediction (distilbert-base-uncased),
adds GPT-style causal text generation (distilgpt2),
compares Zero-Shot vs Few-Shot prompting,
and contrasts in-context prompt learning with full model fine-tuning.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import sys
try:
    import regex
except ImportError:
    import re
    sys.modules["regex"] = re

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from services.llm_service import run_prompt_comparison_demo, call_gemini_api

st.set_page_config(page_title="GPT, BERT & Attention Lab", page_icon="🔬", layout="wide")

st.title("🔬 GPT, BERT & Attention Laboratory")
st.markdown(
    "Explore the architectural evolution of modern Foundation Models: "
    "Autoregressive generation (GPT), Bidirectional representation (BERT), "
    "Dense semantic retrieval (Sentence-BERT), and Cross-modal alignment (CLIP)."
)

# ---------------------------------------------------------------------------
# Helper: load real attention weights from all-MiniLM-L6-v2
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_attn_model():
    """Load the Sentence-BERT encoder with output_attentions=True (cached)."""
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        output_attentions=True,
    )
    model.eval()
    return tokenizer, model


def get_real_attention(sentence: str, head: int = 0):
    """
    Return (tokens, attn_matrix) for a given sentence and attention head.
    Uses all-MiniLM-L6-v2 (6 layers, 12 heads each).
    Returns the LAST layer's chosen head averaged over CLS/SEP context.
    """
    import torch
    tokenizer, model = _load_attn_model()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=24)
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions: tuple of (1, num_heads, seq_len, seq_len) per layer
    last_layer_attn = outputs.attentions[-1]  # shape: [1, 12, seq_len, seq_len]
    attn_head = last_layer_attn[0, head, :, :].numpy()  # [seq_len, seq_len]

    token_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    # Clean up wordpiece tokens for display
    display_tokens = [t.replace("▁", "").replace("##", "·") for t in tokens]
    return display_tokens, attn_head


# ---------------------------------------------------------------------------
# Helper: BERT Masked Language Model prediction
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_fill_mask_pipeline():
    from transformers import pipeline
    return pipeline("fill-mask", model="distilbert-base-uncased")


# ---------------------------------------------------------------------------
# Helper: GPT-style causal text generation
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_gpt_pipeline():
    from transformers import pipeline
    return pipeline(
        "text-generation",
        model="distilgpt2",
        pad_token_id=50256,  # eos_token_id for distilgpt2
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_compare, tab_attn, tab_bert, tab_gpt, tab_prompting, tab_finetune = st.tabs([
    "🏛️ Architectural Comparison",
    "👁️ Real Attention Visualizer",
    "🎭 BERT Masked Prediction",
    "✍️ GPT Text Generation",
    "💬 Prompting Laboratory",
    "⚖️ Prompt-Based vs Fine-Tuning",
])

# ============================================================
# TAB 1 — Architectural Comparison
# ============================================================
with tab_compare:
    st.subheader("1. Generative GPT vs Representational BERT")
    st.markdown(
        """
        | Dimension | GPT Family (e.g. GPT-4, LLaMA) | BERT Family (e.g. RoBERTa, DeBERTa) | Sentence-BERT (MiniLM-L6) | CLIP (ViT + Transformer) |
        | :--- | :--- | :--- | :--- | :--- |
        | **Primary Role** | Open-ended Text Generation | Contextual Word Representation | Dense Sentence Embedding | Multimodal Contrastive Alignment |
        | **Transformer Type** | **Decoder-only** (Causal Attention) | **Encoder-only** (Bidirectional) | **Siamese Dual-Encoder** | **Dual Vision-Language Encoder** |
        | **Attention Mask** | Causal (tokens only attend to prior tokens) | Fully Bidirectional (attends left and right) | Bidirectional + Mean Pooling | Causal (text) + Full (image patches) |
        | **Training Task** | Next-Token Prediction | Masked Language Modeling | Cosine Sim / N-pair Metric Learning | InfoNCE Cross-Modal Contrastive Loss |
        | **Output Dimension**| Token logits over vocabulary (V ≈ 50k) | Contextualized token vectors (768-D) | Fixed 384-D dense vector per sentence | Fixed 512-D vector in shared space |
        | **Best Suited For** | Reasoning, Dialogue, Creative Writing | Classification, Named Entity Recognition | Fast Semantic Search, Clustering | Zero-Shot Image Classification, Retrieval |
        """
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(
            r"""
            #### Causal Autoregressive Factorization (GPT)
            $$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, w_2, \dots, w_{t-1})$$
            The causal mask strictly prevents future information leakage, compelling the network to act as a step-by-step generative system.
            """
        )
    with col_d2:
        st.markdown(
            r"""
            #### Bidirectional Contextual Conditioning (BERT)
            $$P(w_{\text{masked}} \mid w_{\text{unmasked}})$$
            Every token attends simultaneously to all prior and subsequent tokens in the sequence, producing rich contextual representations without autoregressive generation.
            """
        )

# ============================================================
# TAB 2 — REAL Attention Visualizer
# ============================================================
with tab_attn:
    st.subheader("2. Real Multi-Head Self-Attention (all-MiniLM-L6-v2)")
    st.markdown(
        r"""
        The **Scaled Dot-Product Attention** formula computes compatibility weights between queries ($Q$) and keys ($K$), 
        and weights the values ($V$):
        $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

        > **Model:** `sentence-transformers/all-MiniLM-L6-v2` — 6 layers, 12 heads per layer.  
        > Attention weights are **extracted directly from the model** (not simulated).
        """
    )

    sample_sentence = st.text_input(
        "Enter a sentence to inspect real token-to-token attention weights:",
        value="The bank manager approved the loan after inspecting the financial statements.",
        key="attn_sentence",
    )

    head_idx = st.slider("Attention Head (0–11):", min_value=0, max_value=11, value=0)

    if st.button("🔍 Extract Real Attention Weights", key="btn_attn"):
        with st.spinner("Running forward pass through all-MiniLM-L6-v2 …"):
            try:
                display_tokens, attn_matrix = get_real_attention(sample_sentence, head=head_idx)

                fig_attn = go.Figure(
                    data=go.Heatmap(
                        z=attn_matrix,
                        x=display_tokens,
                        y=display_tokens,
                        colorscale="Blues",
                        colorbar=dict(title="Attention Weight"),
                        hoverongaps=False,
                        zmin=0.0,
                        zmax=float(attn_matrix.max()),
                    )
                )
                fig_attn.update_layout(
                    title=f"<b>Real Attention — Layer 6, Head {head_idx}</b> | Model: all-MiniLM-L6-v2",
                    xaxis=dict(title="Keys (Source Tokens)", tickangle=45),
                    yaxis=dict(title="Queries (Target Tokens)", autorange="reversed"),
                    margin=dict(l=60, r=40, t=60, b=80),
                )
                st.plotly_chart(fig_attn, use_container_width=True)
                st.caption(
                    f"Rows = query tokens attending to column key tokens. "
                    f"Extracted from last encoder layer, head {head_idx}. "
                    f"Attention shape: {attn_matrix.shape[0]} × {attn_matrix.shape[1]}."
                )

                # Summary stats
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("Tokens", attn_matrix.shape[0])
                col_s2.metric("Max Attention", f"{attn_matrix.max():.4f}")
                col_s3.metric("Mean Attention", f"{attn_matrix.mean():.4f}")

            except Exception as exc:
                st.error(f"Attention extraction failed: {exc}")
    else:
        st.info("👆 Click the button to extract real attention weights from the model.")

# ============================================================
# TAB 3 — BERT Masked Language Model
# ============================================================
with tab_bert:
    st.subheader("3. BERT Masked Language Model Prediction")
    st.markdown(
        """
        BERT was pre-trained on the **Masked Language Modeling (MLM)** objective.  
        Replace any word in a sentence with `[MASK]` and BERT will predict the most probable completions
        based on **full bidirectional context** (all surrounding tokens).

        > **Model:** `distilbert-base-uncased` (66M parameters, 6-layer distilled BERT)
        """
    )

    masked_input = st.text_input(
        "Enter a sentence containing exactly one `[MASK]` token:",
        value="The capital of France is [MASK].",
        key="bert_mask_input",
    )

    top_k = st.slider("Number of top predictions to show:", min_value=1, max_value=10, value=5)

    if st.button("🎭 Predict Masked Token", key="btn_bert"):
        if "[MASK]" not in masked_input:
            st.warning("Your sentence must contain exactly one `[MASK]` token.")
        else:
            with st.spinner("Running distilbert-base-uncased fill-mask pipeline …"):
                try:
                    fill_mask = _load_fill_mask_pipeline()
                    predictions = fill_mask(masked_input, top_k=top_k)

                    st.success("✅ Predictions from real BERT model:")
                    pred_data = []
                    for rank, pred in enumerate(predictions, start=1):
                        pred_data.append({
                            "Rank": rank,
                            "Token": f"`{pred['token_str'].strip()}`",
                            "Score": f"{pred['score']:.4f}",
                            "Filled Sentence": pred["sequence"],
                        })

                    # Display as markdown table (avoids pandas DLL issues)
                    header = "| Rank | Token | Score | Filled Sentence |\n|---|---|---|---|\n"
                    rows = "\n".join(
                        f"| {r['Rank']} | {r['Token']} | {r['Score']} | {r['Filled Sentence']} |"
                        for r in pred_data
                    )
                    st.markdown(header + rows)

                    # Probability bar chart
                    tokens_list = [p["token_str"].strip() for p in predictions]
                    scores_list = [p["score"] for p in predictions]
                    fig_bar = go.Figure(
                        go.Bar(
                            x=tokens_list,
                            y=scores_list,
                            marker_color="steelblue",
                            text=[f"{s:.3f}" for s in scores_list],
                            textposition="outside",
                        )
                    )
                    fig_bar.update_layout(
                        title="<b>BERT MLM — Top-K Token Probabilities</b>",
                        xaxis_title="Predicted Token",
                        yaxis_title="Probability Score",
                        yaxis_range=[0, min(1.0, max(scores_list) * 1.3)],
                        margin=dict(t=50, b=50),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    st.caption(
                        "Scores are **softmax probabilities** over the full vocabulary. "
                        "BERT uses bidirectional context — all tokens left and right of [MASK] "
                        "influence the prediction simultaneously."
                    )

                except Exception as exc:
                    st.error(f"BERT fill-mask prediction failed: {exc}")
    else:
        st.info("👆 Enter a sentence with `[MASK]` and click Predict.")

# ============================================================
# TAB 4 — GPT-style Text Generation
# ============================================================
with tab_gpt:
    st.subheader("4. GPT-Style Causal Text Generation")
    st.markdown(
        """
        GPT models generate text **left-to-right** using a **causal (unidirectional) decoder**.
        Each new token is sampled from the probability distribution conditioned on all prior tokens.

        > **Local model:** `distilgpt2` (82M parameters, 6-layer distilled GPT-2)  
        > **Gemini API:** If a valid API key is configured, Gemini 2.5 Flash is used for higher quality generation.
        """
    )

    gpt_prompt = st.text_area(
        "Enter a prompt for text generation:",
        value="Generative artificial intelligence is transforming",
        height=80,
        key="gpt_prompt",
    )

    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        max_new_tokens = st.slider("Max new tokens:", min_value=20, max_value=200, value=80)
    with col_g2:
        temperature = st.slider("Temperature:", min_value=0.1, max_value=2.0, value=0.9, step=0.1)
    with col_g3:
        top_p = st.slider("Top-p (nucleus sampling):", min_value=0.1, max_value=1.0, value=0.92, step=0.01)

    use_gemini = st.checkbox(
        "🌟 Use Gemini API (if key configured) instead of distilgpt2",
        value=True,
    )

    if st.button("✍️ Generate Text", key="btn_gpt"):
        if not gpt_prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            generated_text = None
            source_label = ""

            # --- Try Gemini first if checkbox selected ---
            if use_gemini:
                with st.spinner("Calling Gemini API …"):
                    gemini_prompt = (
                        f"Continue the following text naturally in 2–3 sentences:\n\n{gpt_prompt.strip()}"
                    )
                    result = call_gemini_api(gemini_prompt)
                    if result:
                        generated_text = gpt_prompt.strip() + " " + result
                        source_label = "Gemini 2.5 Flash (via API)"

            # --- Fallback: distilgpt2 ---
            if generated_text is None:
                with st.spinner("Running distilgpt2 locally …"):
                    try:
                        gpt_pipe = _load_gpt_pipeline()
                        output = gpt_pipe(
                            gpt_prompt.strip(),
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=True,
                            num_return_sequences=1,
                        )
                        generated_text = output[0]["generated_text"]
                        source_label = "distilgpt2 (local, CPU)"
                    except Exception as exc:
                        st.error(f"GPT generation failed: {exc}")

            if generated_text:
                st.success(f"✅ Generated by **{source_label}**")
                # Highlight prompt vs generated portion
                prompt_len = len(gpt_prompt.strip())
                prompt_part = generated_text[:prompt_len]
                generated_part = generated_text[prompt_len:]

                st.markdown("**Output:**")
                st.markdown(
                    f"<div style='background:#1e1e2e; color:#cdd6f4; padding:14px; "
                    f"border-radius:8px; font-family:monospace; white-space:pre-wrap; line-height:1.6'>"
                    f"<span style='color:#a6e3a1; font-weight:bold'>{prompt_part}</span>"
                    f"<span style='color:#cdd6f4'>{generated_part}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.caption(
                    f"🟢 Green = your prompt | ⬜ White = model-generated continuation | "
                    f"Source: {source_label}"
                )

                # Explain generation parameters
                with st.expander("📐 Decoding Parameters Explained"):
                    st.markdown(
                        f"""
                        | Parameter | Value | Effect |
                        |---|---|---|
                        | **Temperature** | `{temperature}` | Higher → more creative/random; Lower → more deterministic |
                        | **Top-p** | `{top_p}` | Nucleus sampling: sample from smallest vocabulary subset covering p probability mass |
                        | **Max new tokens** | `{max_new_tokens}` | Hard cap on output length |
                        """
                    )
    else:
        st.info("👆 Enter a prompt and click Generate Text.")

# ============================================================
# TAB 5 — Prompting Laboratory
# ============================================================
with tab_prompting:
    st.subheader("5. Prompting Techniques: Zero-Shot vs Few-Shot")
    st.write("Compare prompt formulation strategies on an identical task.")

    test_input = st.text_input(
        "Enter custom review or statement to evaluate:",
        value="The battery life was surprisingly good, but the software crashed twice on startup.",
        key="prompting_input",
    )

    if st.button("🚀 Run Live Prompt Comparison", key="btn_prompt"):
        with st.spinner("Executing prompt comparison via LLM service..."):
            prompt_res = run_prompt_comparison_demo(test_input)

        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.markdown("#### 🅰️ Zero-Shot Prompting")
            st.caption(prompt_res["zero_shot"]["technique"])
            st.code(prompt_res["zero_shot"]["prompt"], language="text")
            st.markdown(f"**Output:** `{prompt_res['zero_shot']['response']}`")
        with c_p2:
            st.markdown("#### 🅱️ Few-Shot Prompting")
            st.caption(prompt_res["few_shot"]["technique"])
            st.code(prompt_res["few_shot"]["prompt"], language="text")
            st.markdown(f"**Output:** `{prompt_res['few_shot']['response']}`")

# ============================================================
# TAB 6 — Prompt-Based Learning vs Fine-Tuning
# ============================================================
with tab_finetune:
    st.subheader("6. Prompt-Based Learning vs Model Fine-Tuning")
    st.markdown(
        """
        | Dimension | Prompt-Based Learning (In-Context) | Parameter-Efficient Fine-Tuning (PEFT / LoRA) | Full Model Fine-Tuning |
        | :--- | :--- | :--- | :--- |
        | **Weights Updated** | **0% (Completely Frozen)** | **0.1% – 2% (Adapter Rank matrices)** | **100% (All Model Parameters)** |
        | **Compute Cost** | Lowest (Standard inference pass) | Moderate (Single GPU / CPU LoRA) | Highest (Multi-GPU clusters) |
        | **Deployment Flexibility** | Universal: Switch tasks via prompt prefix | High: Load lightweight adapters per user | Low: Must host separate model instances |
        | **Catastrophic Forgetting** | None (Base weights untouched) | Minimal | High risk without careful regularization |
        | **Context Length Overhead**| High (Exemplars consume token budget) | None (Prompts remain concise) | None |
        | **Domain Specialization** | Moderate (Constrained by base model priors) | High | Maximum |
        """
    )
