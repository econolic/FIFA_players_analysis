"""model.py - pipeline factory, Optuna optimisation & trainer."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

import joblib
import optuna
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import LOGGER, NUMERIC_COLS, CATEGORICAL_COLS, timeit

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def make_pipeline() -> Tuple[LGBMClassifier, dict]:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(with_mean=False), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=20), CATEGORICAL_COLS),
        ],
        verbose_feature_names_out=False,
    )

    clf = LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Combine preprocessing and classifier
    pipe = LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    space = {
        "n_estimators": (100, 800),
        "max_depth": (5, 20),
        "num_leaves": (31, 255),
        "learning_rate": (0.01, 0.2),
    }
    return pipe, space


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class ModelTrainer:
    """Handles Optuna search, fit, evaluation and explainability."""

    def __init__(self) -> None:
        self.pipeline, self.space = make_pipeline()
        self.study: optuna.Study | None = None
        self.preprocessor = None

    def _validate_classes(self, y: pd.Series) -> None:
        """Validate class sizes and warn about potential issues."""
        class_counts = y.value_counts()
        min_samples = class_counts.min()
        
        if min_samples < 5:
            LOGGER.warning(
                "Some classes have very few samples (min=%d). "
                "Using class weights instead of SMOTE.",
                min_samples
            )
        
        # Log class distribution
        LOGGER.info("Class distribution:\n%s", class_counts)

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to features."""
        if self.preprocessor is None:
            self.preprocessor = ColumnTransformer(
                [
                    ("num", StandardScaler(with_mean=False), NUMERIC_COLS),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=20), CATEGORICAL_COLS),
                ],
                verbose_feature_names_out=False,
            )
            self.preprocessor.fit(X)
        return self.preprocessor.transform(X)

    # ----- optimisation ----- #
    def _objective(self, trial: optuna.trial.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        params = {
            name: trial.suggest_int(name, *rng) if isinstance(rng[0], int)
            else trial.suggest_float(name, *rng, log=True)
            for name, rng in self.space.items()
        }
        self.pipeline.set_params(**params)
        cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
        X_pre = self._preprocess(X)
        scores = cross_val_score(self.pipeline, X_pre, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        return scores.mean()

    class EarlyStoppingCallback:
        def __init__(self, patience: int = 30):
            self.patience = patience
            self.best_value = None
            self.counter = 0

        def __call__(self, study, trial):
            if self.best_value is None or study.best_value > self.best_value:
                self.best_value = study.best_value
                self.counter = 0
            else:
                self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered: no improvement in {self.patience} trials.")
                study.stop()

    @timeit
    def optimise(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 30) -> None:
        self._validate_classes(y)
        early_stopping = self.EarlyStoppingCallback(patience=30)
        self.study = optuna.create_study(direction="maximize", study_name="lgbm_opt")
        self.study.optimize(lambda t: self._objective(t, X, y), n_trials=n_trials, n_jobs=1, callbacks=[early_stopping])
        LOGGER.info("Optuna best F1 = %.3f", self.study.best_value)
        best = {k: v for k, v in self.study.best_params.items()}
        self.pipeline.set_params(**best)

    # ----- fit & evaluate ----- #
    @timeit
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._validate_classes(y)
        X_pre = self._preprocess(X)
        self.pipeline.fit(X_pre, y)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_pre = self._preprocess(X)
        preds = self.pipeline.predict(X_pre)
        LOGGER.info("\n%s", classification_report(y, preds))
        LOGGER.info("Macro F1: %.3f", f1_score(y, preds, average="macro"))

    # ----- explainability ----- #
    def shap_summary(self, X_sample: pd.DataFrame, max_display: int = 15) -> None:
        X_pre = self._preprocess(X_sample)
        explainer = shap.TreeExplainer(self.pipeline)
        shap_values = explainer.shap_values(X_pre, check_additivity=False)
        feature_names = self.preprocessor.get_feature_names_out()
        shap.summary_plot(shap_values, X_pre, feature_names=feature_names, max_display=max_display)

    # ----- persistence ----- #
    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path else Path("models") / f"position_{datetime.now():%Y%m%d_%H%M%S}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((self.pipeline, self.preprocessor), path)
        LOGGER.info("Model saved → %s", path)
        return path

    @staticmethod
    def load(path: Path | str):
        pipeline, preprocessor = joblib.load(path)
        trainer = ModelTrainer()
        trainer.pipeline = pipeline
        trainer.preprocessor = preprocessor
        LOGGER.info("Model loaded from → %s", path)
        return trainer