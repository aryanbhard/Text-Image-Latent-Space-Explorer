"""
PAGE 8: Responsible AI & Bias Analysis (Unit 5 Requirements)
Executes an empirical occupation-gender association bias probe using Sentence-BERT embeddings,
visualizes association differentials, documents ethical limitations, and provides safety notices.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import streamlit as st
from utils.bias_analysis import run_occupation_gender_bias_probe
from utils.visualization import plot_bias_differential_chart, render_table_as_markdown

st.set_page_config(page_title="Responsible AI & Bias Probe", page_icon="⚖️", layout="wide")

st.title("⚖️ Responsible AI & Embedding Bias Probe")
st.markdown(
    "Empirically evaluate semantic associations between occupations and gender-related terminology "
    "in dense pretrained language representations (**Sentence-BERT**)."
)

# Mandatory Safety & Ethical Notice
st.error(
    "🛡️ **MANDATORY RESPONSIBLE AI SAFETY NOTICE (Academic & Ethical Guidelines):**<br>"
    "1. **Mathematical Artifact, Not Individual Fact:** Embedding distances reflect statistical co-occurrence "
    "patterns present within web-scale pretraining corpora (e.g., Wikipedia, Common Crawl). They **do NOT** reflect "
    "innate individual attributes, job competency, or personal identities.<br>"
    "2. **No Real-Person Gender Inference:** This probe strictly evaluates controlled linguistic templates. "
    "It cannot and must never be utilized to infer the gender, competence, or qualifications of real individuals.<br>"
    "3. **Not Proof of Intentional Discrimination:** Embedding associations demonstrate algorithmic representational skew, "
    "not causal or intentional institutional discrimination.",
    icon="🛡️",
)

# Execute Real Bias Probe
with st.spinner("Computing real Sentence-BERT embedding similarities and association differentials..."):
    probe_results = run_occupation_gender_bias_probe()

# Ensure we have a list of dicts for universal display
records = probe_results if isinstance(probe_results, list) else probe_results.to_dict("records")

# Tabs
tab_table, tab_chart, tab_method, tab_licenses = st.tabs([
    "📊 Empirical Bias Probe Table",
    "📈 Association Differential Chart",
    "🔬 Methodology & Critical Limitations",
    "📜 Licences, Attribution & Model Card",
])

with tab_table:
    st.subheader("1. Occupation-Gender Association Scores")
    st.markdown(
        """
        Controlled template: `This person is employed as a {occupation}.`<br>
        - **Male Similarity:** Average cosine similarity to male lexical anchor sentences.<br>
        - **Female Similarity:** Average cosine similarity to female lexical anchor sentences.<br>
        - **Differential (Δ):** $\\Delta = \\text{Sim}(\\text{Female}) - \\text{Sim}(\\text{Male})$.
        """,
        unsafe_allow_html=True,
    )
    
    # Display table without invoking pandas
    st.markdown(render_table_as_markdown(records), unsafe_allow_html=True)

with tab_chart:
    st.subheader("2. Association Differential Distribution")
    st.caption("Positive values indicate higher cosine proximity to female anchors; negative values indicate higher proximity to male anchors.")
    
    fig = plot_bias_differential_chart(records)
    st.plotly_chart(fig, use_container_width=True)

with tab_method:
    st.subheader("3. Methodology & Critical Scientific Limitations")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(
            r"""
            #### Formal Mathematical Formulation
            Let $v_{\text{occ}} \in \mathbb{R}^d$ be the normalized embedding for an occupation sentence:
            $$v_{\text{occ}} = \frac{f(\text{"This person is a \{occ\}"})}{\|f(\text{"This person is a \{occ\}"})\|}$$
            
            Let $\mathbf{c}_M = \frac{1}{|M|} \sum_{m \in M} v_m$ and $\mathbf{c}_F = \frac{1}{|F|} \sum_{w \in F} v_w$ be the centroids of male and female anchor probes.
            
            The **Association Differential** $\Delta$ is defined as:
            $$\Delta = \cos(v_{\text{occ}}, \mathbf{c}_F) - \cos(v_{\text{occ}}, \mathbf{c}_M)$$
            """
        )
    with col_m2:
        st.markdown(
            """
            #### Documented Limitations of Embedding-Based Probes
            1. **Template Sensitivity:** Changing `This person is a...` to `The professional is...` can shift cosine scores by up to 0.04.
            2. **Polysemy & Frequency Skew:** Pretrained models are heavily influenced by word frequency; rarer job titles exhibit nosier embedding trajectories.
            3. **Orthogonality of Metric Spaces:** Small differences in angular cosine values ($\Delta \approx 0.05$) may not translate directly into downstream task discrimination.
            4. **Intersectionality Blindness:** Bipolar gender probes fail to capture intersectional factors such as socio-economic, racial, cultural, or non-binary linguistic diversity.
            """
        )

with tab_licenses:
    st.subheader("4. Open Source Licences & Model Attribution")
    st.markdown(
        """
        | Asset | Origin / Authors | License | Usage in Project |
        | :--- | :--- | :--- | :--- |
        | **Fashion-MNIST Dataset** | Zalando Research (Xiao et al., 2017) | **MIT License** | PyTorch VAE Training & Latent Space Decoding |
        | **Sentence-BERT (`all-MiniLM-L6-v2`)** | UKP Lab / Reimers & Gurevych | **Apache License 2.0** | 384-D Text Embeddings & Bias Probing |
        | **OpenAI CLIP (`clip-vit-base-patch32`)** | OpenAI (Radford et al., 2021) | **MIT License** | 512-D Vision & Multimodal Text Alignment |
        | **PyTorch & Torchvision** | PyTorch Foundation / Meta AI | **BSD-3-Clause** | Deep Learning Core Infrastructure |
        | **Scikit-Learn** | Scikit-Learn Developers | **BSD-3-Clause** | PCA, K-Means Clustering, Nearest Neighbors |
        | **Streamlit Framework** | Snowflake Inc. / Streamlit Team | **Apache License 2.0** | Interactive Web Dashboard |
        | **Plotly Graph Objects** | Plotly Technologies Inc. | **MIT License** | Interactive 2D Visualizations |
        """
    )
