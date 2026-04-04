"""
Baseline models: TF-IDF + RandomForest and TF-IDF + LinearSVC.

Replicates the IGAR (Isnan & Pardamean, 2026) baseline approach for comparison
with the IndoBERT fine-tuned model.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

ModelType = Literal["rf", "svc"]


# ---------------------------------------------------------------------------
# Pipeline Construction
# ---------------------------------------------------------------------------

def build_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Build a TF-IDF vectorizer with project settings.

    Returns
    -------
    TfidfVectorizer
        Configured but unfitted vectorizer.
    """
    return TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b\w+\b",
    )


def build_classifier(model_type: ModelType) -> Any:
    """
    Build the classifier component for the pipeline.

    Parameters
    ----------
    model_type : str
        'rf' for RandomForestClassifier or 'svc' for LinearSVC.

    Returns
    -------
    sklearn estimator
    """
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            n_jobs=-1,
            random_state=config.RANDOM_SEED,
            class_weight="balanced",
        )
    elif model_type == "svc":
        return LinearSVC(
            max_iter=config.SVC_MAX_ITER,
            random_state=config.RANDOM_SEED,
            class_weight="balanced",
            C=1.0,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'rf' or 'svc'.")


def build_pipeline(model_type: ModelType) -> Pipeline:
    """
    Build a full TF-IDF + classifier sklearn pipeline.

    Parameters
    ----------
    model_type : ModelType
        'rf' or 'svc'.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline.
    """
    return Pipeline([
        ("tfidf", build_tfidf_vectorizer()),
        ("clf", build_classifier(model_type)),
    ])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_baseline(
    X_train: List[str],
    y_train: List[int],
    model_type: ModelType = "svc",
) -> Pipeline:
    """
    Fit a TF-IDF + classifier pipeline on training data.

    Parameters
    ----------
    X_train : List[str]
        Preprocessed training texts.
    y_train : List[int]
        Training labels.
    model_type : ModelType
        'rf' or 'svc'.

    Returns
    -------
    Pipeline
        Fitted sklearn pipeline.
    """
    print(f"Training TF-IDF + {model_type.upper()} baseline...")
    pipeline = build_pipeline(model_type)
    pipeline.fit(X_train, y_train)
    print(f"  Done. Vocabulary size: {len(pipeline['tfidf'].vocabulary_):,}")
    return pipeline


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline(
    pipeline: Pipeline,
    X_test: List[str],
    y_test: List[int],
) -> Dict:
    """
    Evaluate a fitted pipeline on test data.

    Computes accuracy, macro F1/precision/recall, per-class metrics, and Cohen's kappa.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn pipeline.
    X_test : List[str]
        Test texts.
    y_test : List[int]
        True labels.

    Returns
    -------
    dict
        Metrics dictionary (mirrors evaluate.compute_metrics output).
    """
    from src.evaluate import compute_metrics

    y_pred = pipeline.predict(X_test)
    return compute_metrics(list(y_test), list(y_pred))


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------

def cross_validate_baseline(
    X: List[str],
    y: List[int],
    model_type: ModelType = "svc",
    n_splits: int = config.CV_N_SPLITS,
) -> Dict:
    """
    Run stratified k-fold cross-validation on the full dataset.

    Parameters
    ----------
    X : List[str]
        All preprocessed texts.
    y : List[int]
        All labels.
    model_type : ModelType
        'rf' or 'svc'.
    n_splits : int
        Number of CV folds.

    Returns
    -------
    dict
        Mean and std of accuracy and macro F1 across folds.
    """
    from sklearn.metrics import make_scorer, f1_score, accuracy_score

    print(f"Running {n_splits}-fold CV for TF-IDF + {model_type.upper()}...")
    pipeline = build_pipeline(model_type)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "macro_f1": make_scorer(f1_score, average="macro", zero_division=0),
    }

    scores = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
    )

    result = {
        "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
        "cv_macro_f1_mean": float(np.mean(scores["test_macro_f1"])),
        "cv_macro_f1_std": float(np.std(scores["test_macro_f1"])),
    }

    print(
        f"  Accuracy: {result['cv_accuracy_mean']:.4f} ± {result['cv_accuracy_std']:.4f} | "
        f"Macro F1: {result['cv_macro_f1_mean']:.4f} ± {result['cv_macro_f1_std']:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# Full Baseline Pipeline
# ---------------------------------------------------------------------------

def run_all_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = config.LABEL_COL,
) -> Dict[str, Dict]:
    """
    Train and evaluate both RF and LinearSVC baselines.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data (must have text_col and label_col).
    test_df : pd.DataFrame
        Test data.
    text_col : str
        Column with preprocessed text.
    label_col : str
        Label column.

    Returns
    -------
    Dict[str, Dict]
        Results keyed by model name ('TF-IDF+RF', 'TF-IDF+SVC').
    """
    X_train = train_df[text_col].tolist()
    y_train = train_df[label_col].tolist()
    X_test = test_df[text_col].tolist()
    y_test = test_df[label_col].tolist()

    results = {}

    for model_type, name in [("rf", "TF-IDF+RF"), ("svc", "TF-IDF+SVC")]:
        pipeline = train_baseline(X_train, y_train, model_type)
        metrics = evaluate_baseline(pipeline, X_test, y_test)
        metrics["model_name"] = name
        results[name] = metrics
        print(
            f"  {name} — Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | "
            f"Kappa: {metrics['cohens_kappa']:.4f}"
        )

    return results


if __name__ == "__main__":
    from src.preprocessing import load_dataset, preprocess_dataframe, remove_duplicates, split_data

    config.set_all_seeds()

    df = load_dataset()
    df = preprocess_dataframe(df)
    df = remove_duplicates(df)
    train_df, val_df, test_df = split_data(df)

    results = run_all_baselines(train_df, test_df)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Macro F1:   {metrics['macro_f1']:.4f}")
        print(f"  Kappa:      {metrics['cohens_kappa']:.4f}")
