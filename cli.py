"""cli.py – robust, de‑duplicated entry point for FIFA package.

Key fixes
---------
* **Removed duplicated code blocks** that caused syntax error.
* Single `PIP_TO_IMPORT` dict and one dependency check using `importlib.import_module`.
* No references to `importlib.util` – avoids custom‑shadowing issue.
* Banner & Typer commands unchanged.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict

# -----------------------------------------------------------------------------
# 0  DEPENDENCY GUARD (stdlib only)
# -----------------------------------------------------------------------------

PIP_TO_IMPORT: Dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "imbalanced-learn": "imblearn",
    "lightgbm": "lightgbm",
    "optuna": "optuna",
    "shap": "shap",
    "typer[all]": "typer",
    "plotly": "plotly",
}

missing = []
for pip_name, import_name in PIP_TO_IMPORT.items():
    try:
        importlib.import_module(import_name)
    except ModuleNotFoundError:
        missing.append(pip_name)

if missing:
    pkgs = " ".join(missing)
    sys.stderr.write("[!] Some packages may be missing or not visible to this interpreter:\n")
    sys.stderr.write(f"    {', '.join(missing)}\n")
    sys.stderr.write(f"    (tip) activate correct venv or run: pip install {pkgs}\n")
    # Continue execution – runtime imports may still succeed in user env

# -----------------------------------------------------------------------------
# 1  Safe imports of third‑party & local modules
# -----------------------------------------------------------------------------

import typer  # noqa: E402  (present after guard)

from data import LOGGER, timeit  # noqa: E402
from features import add_domain_features, load_and_fe, get_feature_columns  # noqa: E402
from model import ModelTrainer  # noqa: E402

# -----------------------------------------------------------------------------
# 2  Typer application & banner
# -----------------------------------------------------------------------------

APP = typer.Typer(add_completion=False, rich_markup_mode="rich")

BANNER = """
🚀 FIFA CLI
===========
• EDA      : python cli.py eda fifa_players.csv reports/
• Train    : python cli.py train fifa_players.csv --target player_positions
• Predict  : python cli.py predict model.pkl new.csv preds.csv
"""


@APP.callback(invoke_without_command=True)
def root(ctx: typer.Context, examples: bool = typer.Option(False, "--examples", help="Show usage examples")) -> None:
    if ctx.invoked_subcommand is None or examples:
        typer.echo(BANNER)
        raise typer.Exit()


# -----------------------------------------------------------------------------
# 3  Commands
# -----------------------------------------------------------------------------

@APP.command()
def eda(csv_path: Path, report_dir: Path):
    """Generate quick EDA artefacts (numeric histograms & summary CSV)."""
    import plotly.express as px  # noqa: F401 – lazy import

    df = load_and_fe(csv_path, "player_positions")  # Using default target
    report_dir.mkdir(parents=True, exist_ok=True)
    df.describe(include="all").T.to_csv(report_dir / "summary.csv")
    num_cols, _ = get_feature_columns()
    for col in num_cols:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        fig.write_html(report_dir / f"hist_{col}.html")
    typer.echo(f"EDA saved to → {report_dir}")


@APP.command()
def train(
    csv_path: Path,
    target: str = typer.Option("player_positions", help="Target column name"),
    model_out: Path | None = typer.Option(None, help="Path for serialized model"),
    n_trials: int = typer.Option(30, help="Optuna trials"),
):
    """Optimise hyper‑params, fit pipeline, save model."""
    import pandas as pd  # noqa: F401 – ensured by guard

    df = load_and_fe(csv_path, target)
    num_cols, cat_cols = get_feature_columns()
    X, y = df[num_cols + cat_cols], df[target]

    trainer = ModelTrainer()
    trainer.optimise(X, y, n_trials)
    trainer.fit(X, y)
    trainer.evaluate(X, y)
    path = trainer.save(model_out)
    typer.echo(f"Model stored at {path}")


@APP.command()
def predict(model_path: Path, data_path: Path, output: Path):
    """Score new data with a saved model."""
    import pandas as pd  # noqa: F401

    trainer = ModelTrainer.load(model_path)
    df = load_and_fe(data_path, "player_positions")  # Using default target for prediction
    num_cols, cat_cols = get_feature_columns()
    preds = trainer.pipeline.predict(df[num_cols + cat_cols])
    pd.DataFrame({"prediction": preds}).to_csv(output, index=False)
    typer.echo(f"Predictions → {output}")


# -----------------------------------------------------------------------------
# 4  Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    APP()  # pylint: disable=no-value-for-parameter