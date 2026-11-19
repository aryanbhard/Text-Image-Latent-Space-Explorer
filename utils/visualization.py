"""
Interactive Plotly visualisations for latent spaces, embeddings, clustering, and bias probes.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple


def plot_embedding_scatter_2d(
    coords_2d: np.ndarray,
    labels: List[str],
    hover_texts: List[str],
    color_indices: Optional[List[int]] = None,
    color_names: Optional[List[str]] = None,
    title: str = "2D Latent Representation",
    reduction_method: str = "PCA",
    highlight_idx: Optional[int] = None,
    neighbor_indices: Optional[List[int]] = None,
) -> go.Figure:
    """
    Create an interactive 2D scatter plot using Plotly.
    Supports point selection, cluster coloring, and nearest neighbor highlighting.
    """
    fig = go.Figure()

    n = len(coords_2d)
    if n == 0:
        fig.update_layout(title="No data points available to plot.")
        return fig

    # Group points by cluster/category if available
    unique_groups = sorted(list(set(color_names))) if color_names else ["All Samples"]

    for grp in unique_groups:
        if color_names:
            indices = [i for i, c in enumerate(color_names) if c == grp]
        else:
            indices = list(range(n))

        x_vals = coords_2d[indices, 0]
        y_vals = coords_2d[indices, 1]
        hover_info = [
            f"<b>Index:</b> {i}<br><b>Label:</b> {labels[i]}<br><b>Content:</b> {hover_texts[i][:120]}..."
            for i in indices
        ]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                name=grp,
                marker=dict(size=11, opacity=0.85, line=dict(width=1, color="DarkSlateGrey")),
                text=hover_info,
                hoverinfo="text",
                customdata=indices,
            )
        )

    # If a query point is selected, highlight it distinctly
    if highlight_idx is not None and 0 <= highlight_idx < n:
        hx = float(coords_2d[highlight_idx, 0])
        hy = float(coords_2d[highlight_idx, 1])
        fig.add_trace(
            go.Scatter(
                x=[hx],
                y=[hy],
                mode="markers+text",
                name="★ Selected Point",
                text=[f"Point #{highlight_idx}"],
                textposition="top center",
                marker=dict(size=18, color="gold", symbol="star", line=dict(width=2, color="black")),
                hoverinfo="text",
                hovertext=f"<b>Selected Query:</b> #{highlight_idx}<br>{hover_texts[highlight_idx]}",
            )
        )

    # If nearest neighbors are provided, highlight them with green rings and connector lines
    if neighbor_indices and highlight_idx is not None and 0 <= highlight_idx < n:
        hx = float(coords_2d[highlight_idx, 0])
        hy = float(coords_2d[highlight_idx, 1])
        nn_x = [coords_2d[idx, 0] for idx in neighbor_indices if 0 <= idx < n]
        nn_y = [coords_2d[idx, 1] for idx in neighbor_indices if 0 <= idx < n]

        # Add connector lines
        for nx, ny in zip(nn_x, nn_y):
            fig.add_trace(
                go.Scatter(
                    x=[hx, nx],
                    y=[hy, ny],
                    mode="lines",
                    line=dict(color="rgba(46, 204, 113, 0.6)", width=1.5, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=nn_x,
                y=nn_y,
                mode="markers",
                name="Nearest Neighbors",
                marker=dict(size=15, color="rgba(46, 204, 113, 0.8)", symbol="circle-open", line=dict(width=3, color="#2ecc71")),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b> ({reduction_method})", font=dict(size=17, color="#2c3e50")),
        xaxis=dict(title=f"{reduction_method} Dimension 1", zeroline=True, gridcolor="#f0f2f6"),
        yaxis=dict(title=f"{reduction_method} Dimension 2", zeroline=True, gridcolor="#f0f2f6"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    return fig


def plot_combined_multimodal_scatter(
    text_coords: np.ndarray,
    text_labels: List[str],
    image_coords: np.ndarray,
    image_labels: List[str],
    reduction_method: str = "PCA",
    is_shared_clip_space: bool = False,
) -> go.Figure:
    """
    Multimodal 2D visualization of text and image embeddings.
    Clearly distinguishes modality and provides accurate technical framing.
    """
    fig = go.Figure()

    # Text points trace
    if len(text_coords) > 0:
        fig.add_trace(
            go.Scatter(
                x=text_coords[:, 0],
                y=text_coords[:, 1],
                mode="markers",
                name="📝 Text Modality",
                marker=dict(size=11, color="#2980b9", symbol="circle", line=dict(width=1, color="white")),
                text=[f"<b>[TEXT]</b> {lbl[:100]}..." for lbl in text_labels],
                hoverinfo="text",
            )
        )

    # Image points trace
    if len(image_coords) > 0:
        fig.add_trace(
            go.Scatter(
                x=image_coords[:, 0],
                y=image_coords[:, 1],
                mode="markers",
                name="🖼️ Image Modality",
                marker=dict(size=13, color="#e67e22", symbol="diamond", line=dict(width=1, color="black")),
                text=[f"<b>[IMAGE]</b> {lbl}" for lbl in image_labels],
                hoverinfo="text",
            )
        )

    sub = (
        "Shared Joint CLIP Space (Genuine Cross-Modal Semantic Proximity)"
        if is_shared_clip_space
        else "Sentence-BERT & Independent Visual Projections (Categorical Modality Mapping)"
    )

    fig.update_layout(
        title=dict(text=f"<b>Multimodal Latent Space Projection</b><br><sup>{sub}</sup>", font=dict(size=16)),
        xaxis=dict(title=f"{reduction_method} Dim 1", gridcolor="#f0f2f6"),
        yaxis=dict(title=f"{reduction_method} Dim 2", gridcolor="#f0f2f6"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=40, t=65, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_vae_latent_scatter(
    latent_coords: np.ndarray,
    class_indices: List[int],
    class_names: List[str],
    selected_coord: Optional[Tuple[float, float]] = None,
) -> go.Figure:
    """
    Plot 2D VAE latent space colored by ground-truth Fashion-MNIST class.
    Shows the continuous Gaussian manifold and allows interactive coordinate inspection.
    """
    fig = go.Figure()

    unique_classes = sorted(list(set(class_indices)))
    for c_idx in unique_classes:
        mask = [i for i, c in enumerate(class_indices) if c == c_idx]
        name = class_names[c_idx] if c_idx < len(class_names) else f"Class {c_idx}"
        fig.add_trace(
            go.Scatter(
                x=latent_coords[mask, 0],
                y=latent_coords[mask, 1],
                mode="markers",
                name=name,
                marker=dict(size=7, opacity=0.75),
                hoverinfo="name+x+y",
            )
        )

    # Highlight currently selected decoding point
    if selected_coord is not None:
        sx, sy = selected_coord
        fig.add_trace(
            go.Scatter(
                x=[sx],
                y=[sy],
                mode="markers+text",
                name="🎯 Selected (z1, z2)",
                text=[f"({sx:.2f}, {sy:.2f})"],
                textposition="top center",
                marker=dict(size=17, color="#e74c3c", symbol="x", line=dict(width=3, color="black")),
            )
        )

    fig.update_layout(
        title=dict(text="<b>Fashion-MNIST 2D VAE Latent Space</b> (z₁, z₂ Manifold)", font=dict(size=16)),
        xaxis=dict(title="Latent Dimension z₁ (Encoder Mean μ₁)", zeroline=True, gridcolor="#f0f2f6"),
        yaxis=dict(title="Latent Dimension z₂ (Encoder Mean μ₂)", zeroline=True, gridcolor="#f0f2f6"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_training_loss_curves(
    epochs: List[int],
    total_losses: List[float],
    bce_losses: List[float],
    kld_losses: List[float],
) -> go.Figure:
    """Plot VAE training loss curves (Total ELBO loss, Reconstruction BCE, and KL Divergence)."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=total_losses,
            mode="lines+markers",
            name="Total Loss (ELBO)",
            line=dict(color="#2980b9", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=bce_losses,
            mode="lines+markers",
            name="Reconstruction Loss (BCE)",
            line=dict(color="#27ae60", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=kld_losses,
            mode="lines+markers",
            name="KL Divergence (KLD)",
            line=dict(color="#e74c3c", width=2, dash="dot"),
        )
    )

    fig.update_layout(
        title=dict(text="<b>VAE Training Convergence Curves</b> (Fashion-MNIST)", font=dict(size=16)),
        xaxis=dict(title="Epoch", gridcolor="#f0f2f6"),
        yaxis=dict(title="Loss Magnitude", gridcolor="#f0f2f6"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_table_as_markdown(data: List[Dict[str, Any]]) -> str:
    """
    Render a list of dictionaries as a GitHub-flavored Markdown table.
    Bypasses pandas completely to avoid Windows Application Control policy restrictions.
    """
    if not data:
        return "*No records available.*"
    headers = list(data[0].keys())
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    separator_row = "| " + " | ".join([":---"] * len(headers)) + " |"
    data_rows = []
    for row in data:
        data_rows.append("| " + " | ".join(str(row.get(h, "")).replace("\n", " ") for h in headers) + " |")
    return "\n".join([header_row, separator_row] + data_rows)


def plot_bias_differential_chart(bias_records: List[Dict[str, Any]]) -> go.Figure:
    """
    Horizontal bar chart depicting occupation association differentials (Female Sim - Male Sim).
    Color-coded: Purple for female association, Orange for male association, Gray for neutral.
    """
    occupations = [r["Occupation"] for r in bias_records]
    differentials = [r["Association Differential (F - M)"] for r in bias_records]

    colors = [
        "#9b59b6" if d > 0.01 else ("#e67e22" if d < -0.01 else "#95a5a6")
        for d in differentials
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=occupations,
            x=differentials,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{d:+.4f}" for d in differentials],
            textposition="auto",
            hoverinfo="y+x",
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>Occupation-Gender Association Differential</b><br><sup>Delta = Sim(Female Anchor) - Sim(Male Anchor) in Sentence-BERT Embedding Space</sup>",
            font=dict(size=15),
        ),
        xaxis=dict(
            title="Association Differential Δ (Positive: Female Bias, Negative: Male Bias)",
            zeroline=True,
            zerolinecolor="#2c3e50",
            zerolinewidth=2,
            gridcolor="#f0f2f6",
        ),
        yaxis=dict(title="Occupation", autorange="reversed"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=150, r=40, t=65, b=40),
    )
    return fig
