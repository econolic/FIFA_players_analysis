"""data.py - I/O & utility helpers for the FIFA position-classifier package.

This module is intentionally lean: it only handles CSV reading, base dtype
mapping, and a `timeit` decorator so other modules can import it without
circular dependencies.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# Logger (shared across package)
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("fifa")
if not LOGGER.handlers:  # avoid duplicate handlers when re‑importing in tests
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DTYPE_MAP: dict[str, str] = {
    "age": "int8",
    "height_cm": "int16",
    "weight_kg": "int16",
    "overall": "int8",
    "potential": "int8",
    "nationality_name": "category",
    "club_name": "category",
    "preferred_foot": "category",
}

NUMERIC_COLS: List[str] = [
    "age",
    "height_cm",
    "weight_kg",
    "overall",
    "potential",
]
CATEGORICAL_COLS: List[str] = [
    "nationality_name",
    "club_name",
    "preferred_foot",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timeit(func):  # type: ignore
    """Simple decorator printing run‑time via shared LOGGER."""

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        LOGGER.info("%s finished in %.2f s", func.__name__, time.perf_counter() - start)
        return result

    return wrapper


def validate_data(df: pd.DataFrame, target_col: str) -> None:
    """Validate data quality and required columns."""
    # Check required columns
    required_cols = NUMERIC_COLS + CATEGORICAL_COLS + [target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for duplicates
    if df.duplicated().any():
        LOGGER.warning("Found %d duplicate rows", df.duplicated().sum())
    
    # Check target column
    if df[target_col].isna().any():
        raise ValueError(f"Target column '{target_col}' contains missing values")
    
    # Check numeric columns
    for col in NUMERIC_COLS:
        if df[col].isna().any():
            LOGGER.warning("Column '%s' contains missing values", col)
    
    # Check categorical columns
    for col in CATEGORICAL_COLS:
        if df[col].isna().any():
            LOGGER.warning("Column '%s' contains missing values", col)


@timeit
def load_data(path: Path | str, target_col: str) -> pd.DataFrame:
    """Load CSV with explicit dtypes and basic sanity filters."""
    # First load without categorical dtypes to handle missing values
    df = pd.read_csv(path, low_memory=False)
    
    # Basic data cleaning
    df = df.dropna(subset=[target_col])  # Remove rows with missing target
    df = df[(df["height_cm"].between(140, 220)) & (df["weight_kg"].between(45, 120))]
    
    # Fill missing values
    for col in NUMERIC_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    for col in CATEGORICAL_COLS:
        if df[col].isna().any():
            # Convert to string first to handle missing values
            df[col] = df[col].astype(str)
            df[col] = df[col].replace("nan", "Unknown")
            # Now convert to category
            df[col] = df[col].astype("category")
    
    # Convert numeric columns to their proper dtypes
    for col, dtype in DTYPE_MAP.items():
        if col in NUMERIC_COLS:
            df[col] = df[col].astype(dtype)
    
    # Validate data
    validate_data(df, target_col)
    
    return df.reset_index(drop=True)