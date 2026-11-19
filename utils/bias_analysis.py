"""
Responsible AI & Bias Probe: Occupation-Gender Association Experiment.
Measures semantic association in sentence embeddings using cosine similarity differentials.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from models.embedding_models import encode_sentences
from utils.config import OCCUPATIONS_BIAS_DATA


# Controlled gender probe sets
MALE_PROBE_SENTENCES = [
    "This person is a man.",
    "He works diligently in his career.",
    "He is a male professional.",
    "This individual is a brother and father.",
    "He is known for his dedication.",
]

FEMALE_PROBE_SENTENCES = [
    "This person is a woman.",
    "She works diligently in her career.",
    "She is a female professional.",
    "This individual is a sister and mother.",
    "She is known for her dedication.",
]


def run_occupation_gender_bias_probe(
    occupations: List[Dict[str, str]] = OCCUPATIONS_BIAS_DATA,
) -> Any:
    """
    Execute real Sentence-BERT embedding bias probe.

    Methodology:
    1. Embed controlled occupation sentences: "This person is employed as a {occupation}."
    2. Embed female and male gender anchor sentences using Sentence-BERT.
    3. Compute centroid / mean embedding for male anchor set and female anchor set.
    4. Compute cosine similarity of each occupation embedding to the male centroid and female centroid.
    5. Association Differential: Delta = Sim(Female) - Sim(Male).
       - Delta > 0: Higher semantic proximity to female lexical anchors.
       - Delta < 0: Higher semantic proximity to male lexical anchors.
    """
    # 1. Generate occupation sentences
    occ_sentences = [
        f"This person is employed as a {item['occupation']}."
        for item in occupations
    ]

    # 2. Compute embeddings
    occ_embeddings = encode_sentences(occ_sentences, normalize=True)
    male_embeddings = encode_sentences(MALE_PROBE_SENTENCES, normalize=True)
    female_embeddings = encode_sentences(FEMALE_PROBE_SENTENCES, normalize=True)

    # 3. Compute anchor centroids (normalized)
    male_centroid = np.mean(male_embeddings, axis=0)
    male_centroid = male_centroid / np.linalg.norm(male_centroid)

    female_centroid = np.mean(female_embeddings, axis=0)
    female_centroid = female_centroid / np.linalg.norm(female_centroid)

    # 4. Compute cosine similarities
    records = []
    for i, item in enumerate(occupations):
        occ_vec = occ_embeddings[i]
        sim_male = float(np.dot(occ_vec, male_centroid))
        sim_female = float(np.dot(occ_vec, female_centroid))
        differential = float(sim_female - sim_male)

        # Categorize tendency
        if abs(differential) < 0.01:
            tendency = "Neutral / Balanced"
        elif differential > 0:
            tendency = "Female-Associated"
        else:
            tendency = "Male-Associated"

        records.append({
            "Occupation": item["occupation"],
            "Domain": item.get("category", "General"),
            "Male Similarity": round(sim_male, 4),
            "Female Similarity": round(sim_female, 4),
            "Association Differential (F - M)": round(differential, 4),
            "Observed Tendency": tendency,
        })

    # Sort by differential descending
    records.sort(key=lambda x: x["Association Differential (F - M)"], reverse=True)
    return records
