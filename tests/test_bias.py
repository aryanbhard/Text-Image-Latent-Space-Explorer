"""
Unit tests for the Responsible AI occupation-gender bias probe.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.bias_analysis import run_occupation_gender_bias_probe


def test_bias_probe_calculation():
    test_occupations = [
        {"occupation": "Software Engineer", "category": "STEM"},
        {"occupation": "Registered Nurse", "category": "Healthcare"},
    ]
    results = run_occupation_gender_bias_probe(test_occupations)

    # Handle both list of dicts and DataFrame
    records = results if isinstance(results, list) else results.to_dict("records")
    assert len(records) == 2, f"Expected 2 records, got {len(records)}"

    for rec in records:
        assert "Occupation" in rec
        assert "Male Similarity" in rec
        assert "Female Similarity" in rec
        assert "Association Differential (F - M)" in rec
        assert "Observed Tendency" in rec
        assert -1.0 <= rec["Male Similarity"] <= 1.0
        assert -1.0 <= rec["Female Similarity"] <= 1.0
        # Check differential formula
        expected_diff = round(rec["Female Similarity"] - rec["Male Similarity"], 4)
        assert abs(rec["Association Differential (F - M)"] - expected_diff) < 1e-3
