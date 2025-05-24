"""features.py - domain feature engineering for FIFA dataset."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from data import LOGGER, NUMERIC_COLS, CATEGORICAL_COLS, load_data, timeit

__all__ = [
    "add_domain_features",
    "get_feature_columns",
    "load_and_fe",
]


@timeit
def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific features; returns new DataFrame copy."""
    df = df.copy()
    
    # Physical features
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
    
    # Experience features
    if "club_joined" in df.columns:
        joined = pd.to_datetime(df["club_joined"], errors="coerce")
        df["club_tenure"] = ((pd.Timestamp("today") - joined).dt.days / 365.25).fillna(0)
    else:
        df["club_tenure"] = 0.0
    
    # Performance features
    if "overall" in df.columns and "potential" in df.columns:
        df["potential_gap"] = df["potential"] - df["overall"]
    
    # Position features
    if "player_positions" in df.columns:
        df["num_positions"] = df["player_positions"].str.count(",") + 1
    
    # ensure new cols appear in shared lists exactly once
    new_numeric = ["bmi", "club_tenure", "potential_gap", "num_positions"]
    for col in new_numeric:
        if col not in NUMERIC_COLS and col in df.columns:
            NUMERIC_COLS.append(col)
    
    return df


def get_feature_columns():
    """Return (numeric, categorical) lists - mutated by FE."""
    return NUMERIC_COLS.copy(), CATEGORICAL_COLS.copy()


def load_and_fe(path: Path | str, target_col: str) -> pd.DataFrame:
    """Load data and apply feature engineering."""
    df = load_data(path, target_col)
    return add_domain_features(df)