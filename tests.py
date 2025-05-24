"""tests.py - minimal pytest suite for fifa package."""
from pathlib import Path

import pandas as pd

from features import add_domain_features, get_feature_columns
from model import ModelTrainer


def test_feature_engineering():
    sample = pd.DataFrame({"weight_kg": [75], "height_cm": [180]})
    enriched = add_domain_features(sample)
    expected_bmi = 75 / (1.8 ** 2)
    assert abs(enriched["bmi"].iloc[0] - expected_bmi) < 1e-3


def test_model_trainer(tmp_path: Path):
    df = pd.DataFrame({
        "age": [21, 22, 23, 24, 25],
        "height_cm": [180, 182, 178, 185, 181],
        "weight_kg": [75, 78, 74, 80, 76],
        "overall": [70, 72, 74, 76, 78],
        "potential": [80, 82, 84, 86, 88],
        "nationality": ["A", "B", "A", "B", "A"],
        "club_name": ["X", "Y", "X", "Y", "X"],
        "preferred_foot": ["Right", "Left", "Right", "Left", "Right"],
        "pos": ["GK", "ST", "GK", "ST", "GK"],
    })
    trainer = ModelTrainer()
    num_cols, cat_cols = get_feature_columns()
    X, y = df[num_cols + cat_cols], df["pos"]
    trainer.optimise(X, y, n_trials=2)
    trainer.fit(X, y)
    model_path = tmp_path / "test.pkl"
    trainer.save(model_path)
    assert model_path.exists()
