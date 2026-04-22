"""
Evaluation metrics and reporting for sentiment classification models.

Computes: accuracy, macro F1/precision/recall, per-class metrics, Cohen's kappa,
          bootstrap confidence intervals, pairwise significance tests.
Produces: confusion matrix plots, comparison tables.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : List[int]
        True labels.
    y_pred : List[int]
        Predicted labels.

    Returns
    -------
    dict
        Keys:
        - accuracy (float)
        - macro_f1 (float)
        - macro_precision (float)
        - macro_recall (float)
        - cohens_kappa (float)
        - per_class (Dict[str, Dict[str, float]])
            e.g. {'Negative': {'f1': ..., 'precision': ..., 'recall': ...}}
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_precision = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(y_true_arr, y_pred_arr))

    # Per-class
    labels_present = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))
    per_class_f1 = f1_score(y_true_arr, y_pred_arr, labels=labels_present, average=None, zero_division=0)
    per_class_prec = precision_score(y_true_arr, y_pred_arr, labels=labels_present, average=None, zero_division=0)
    per_class_rec = recall_score(y_true_arr, y_pred_arr, labels=labels_present, average=None, zero_division=0)

    per_class = {}
    for i, label_id in enumerate(labels_present):
        label_name = config.LABEL_MAP.get(label_id, str(label_id))
        per_class[label_name] = {
            "f1": float(per_class_f1[i]),
            "precision": float(per_class_prec[i]),
            "recall": float(per_class_rec[i]),
        }

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "cohens_kappa": kappa,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def classification_report_df(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate a per-class classification report as a DataFrame.

    Parameters
    ----------
    y_true : List[int]
        True labels.
    y_pred : List[int]
        Predicted labels.
    labels : List[str], optional
        Class names. Defaults to config.LABEL_NAMES.

    Returns
    -------
    pd.DataFrame
        Rows: class names + macro avg + weighted avg.
        Columns: precision, recall, f1-score, support.
    """
    _labels = labels or config.LABEL_NAMES
    report_str = classification_report(
        y_true, y_pred,
        target_names=_labels,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report_str).T
    df["support"] = df["support"].astype(int)
    return df


def compare_models_table(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build a comparison DataFrame across multiple models.

    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Keys are model names, values are outputs of compute_metrics().

    Returns
    -------
    pd.DataFrame
        Rows: models, columns: accuracy, macro_f1, macro_precision, macro_recall, cohens_kappa.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        row = {
            "Model": model_name,
            "Accuracy": metrics.get("accuracy", float("nan")),
            "Macro F1": metrics.get("macro_f1", float("nan")),
            "Macro Precision": metrics.get("macro_precision", float("nan")),
            "Macro Recall": metrics.get("macro_recall", float("nan")),
            "Cohen's Kappa": metrics.get("cohens_kappa", float("nan")),
        }
        # Add per-class F1
        for cls_name in config.LABEL_NAMES:
            pc = metrics.get("per_class", {}).get(cls_name, {})
            row[f"F1 ({cls_name})"] = pc.get("f1", float("nan"))
        rows.append(row)

    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot and optionally save a confusion matrix heatmap.

    Parameters
    ----------
    y_true : List[int]
        True labels.
    y_pred : List[int]
        Predicted labels.
    labels : List[str], optional
        Class names for axis ticks.
    save_path : str or Path, optional
        If provided, saves the figure to this path.
    normalize : bool
        If True, normalize by true counts (show rates, not counts).
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _labels = labels or config.LABEL_NAMES
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(_labels))))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_display = cm
        fmt = "d"
        vmax = None

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=_labels,
        yticklabels=_labels,
        ax=ax,
        vmin=0,
        vmax=vmax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=config.PLOT_DPI, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Model Comparison Plot
# ---------------------------------------------------------------------------

def plot_performance_comparison(
    results_dict: Dict[str, Dict],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing key metrics across models.

    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Keyed by model name, values from compute_metrics().
    save_path : str or Path, optional
        Save path for the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    metrics_to_plot = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "cohens_kappa"]
    metric_labels = ["Accuracy", "Macro F1", "Precision", "Recall", "Kappa"]

    models = list(results_dict.keys())
    x = np.arange(len(metric_labels))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, (model_name, color) in enumerate(zip(models, colors)):
        values = [results_dict[model_name].get(m, 0.0) for m in metrics_to_plot]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width * 0.9, label=model_name, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=13)
    ax.legend(loc="upper right")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=config.PLOT_DPI, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Bootstrap Confidence Intervals & Significance Testing
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: List[int],
    y_pred: List[int],
    metric_fn=None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute bootstrap confidence intervals for classification metrics.

    Parameters
    ----------
    y_true : List[int]
        True labels.
    y_pred : List[int]
        Predicted labels.
    metric_fn : callable, optional
        Function(y_true, y_pred) -> float. Defaults to macro-F1.
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (e.g. 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: 'accuracy', 'macro_f1', 'cohens_kappa'.
        Values: (point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n = len(y_true_arr)

    alpha = (1.0 - confidence) / 2.0

    boot_scores = {"accuracy": [], "macro_f1": [], "cohens_kappa": []}

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true_arr[idx]
        yp = y_pred_arr[idx]
        boot_scores["accuracy"].append(accuracy_score(yt, yp))
        boot_scores["macro_f1"].append(f1_score(yt, yp, average="macro", zero_division=0))
        boot_scores["cohens_kappa"].append(cohen_kappa_score(yt, yp))

    results = {}
    for metric_name, scores in boot_scores.items():
        scores_arr = np.array(scores)
        point_est = {
            "accuracy": accuracy_score(y_true_arr, y_pred_arr),
            "macro_f1": f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0),
            "cohens_kappa": cohen_kappa_score(y_true_arr, y_pred_arr),
        }[metric_name]
        ci_lower = float(np.percentile(scores_arr, 100 * alpha))
        ci_upper = float(np.percentile(scores_arr, 100 * (1.0 - alpha)))
        results[metric_name] = (point_est, ci_lower, ci_upper)

    return results


def bootstrap_paired_test(
    y_true: List[int],
    y_pred_a: List[int],
    y_pred_b: List[int],
    metric: str = "macro_f1",
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap-based paired significance test between two models.

    Tests H0: metric(model_A) >= metric(model_B) against
         H1: metric(model_A) < metric(model_B).

    Parameters
    ----------
    y_true : List[int]
        True labels.
    y_pred_a : List[int]
        Predictions from model A (e.g. baseline).
    y_pred_b : List[int]
        Predictions from model B (e.g. IndoBERT).
    metric : str
        One of 'macro_f1', 'accuracy', 'cohens_kappa'.
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: 'metric_a', 'metric_b', 'delta', 'p_value',
              'significant_at_0.05', 'n_bootstrap'.
    """
    rng = np.random.RandomState(seed)
    y_true_arr = np.array(y_true)
    ya = np.array(y_pred_a)
    yb = np.array(y_pred_b)
    n = len(y_true_arr)

    def _score(yt, yp, m):
        if m == "macro_f1":
            return f1_score(yt, yp, average="macro", zero_division=0)
        elif m == "accuracy":
            return accuracy_score(yt, yp)
        elif m == "cohens_kappa":
            return cohen_kappa_score(yt, yp)
        else:
            raise ValueError(f"Unknown metric: {m}")

    observed_a = _score(y_true_arr, ya, metric)
    observed_b = _score(y_true_arr, yb, metric)
    observed_delta = observed_b - observed_a

    count_ge_zero = 0
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true_arr[idx]
        delta_boot = _score(yt, yb[idx], metric) - _score(yt, ya[idx], metric)
        if delta_boot >= 0:
            count_ge_zero += 1

    p_value = 1.0 - count_ge_zero / n_bootstrap

    return {
        "metric": metric,
        "metric_a": float(observed_a),
        "metric_b": float(observed_b),
        "delta": float(observed_delta),
        "p_value": float(p_value),
        "significant_at_0.05": p_value < 0.05,
        "n_bootstrap": n_bootstrap,
    }


def significance_comparison_table(
    y_true: List[int],
    model_predictions: Dict[str, List[int]],
    baseline_name: str = None,
    metrics: List[str] = None,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run pairwise bootstrap significance tests for all models vs. a baseline.

    Parameters
    ----------
    y_true : List[int]
        True labels.
    model_predictions : Dict[str, List[int]]
        Model name → predictions.
    baseline_name : str, optional
        Baseline model name. If None, uses the first model in the dict.
    metrics : List[str], optional
        Metrics to test. Defaults to ['macro_f1', 'accuracy', 'cohens_kappa'].
    n_bootstrap : int
    seed : int

    Returns
    -------
    pd.DataFrame
        Columns: model, metric, baseline_score, model_score, delta, p_value, significant.
    """
    _metrics = metrics or ["macro_f1", "accuracy", "cohens_kappa"]
    _baseline = baseline_name or list(model_predictions.keys())[0]
    baseline_preds = model_predictions[_baseline]

    rows = []
    for model_name, preds in model_predictions.items():
        if model_name == _baseline:
            continue
        for m in _metrics:
            result = bootstrap_paired_test(
                y_true, baseline_preds, preds,
                metric=m, n_bootstrap=n_bootstrap, seed=seed,
            )
            rows.append({
                "comparison": f"{model_name} vs {_baseline}",
                "metric": m,
                "baseline_score": result["metric_a"],
                "model_score": result["metric_b"],
                "delta": result["delta"],
                "p_value": result["p_value"],
                "significant_0.05": result["significant_at_0.05"],
            })

    return pd.DataFrame(rows)


def bootstrap_ci_table(
    model_ci_results: Dict[str, Dict[str, Tuple[float, float, float]]],
) -> pd.DataFrame:
    """
    Format bootstrap CI results into a publication-ready DataFrame.

    Parameters
    ----------
    model_ci_results : dict
        {model_name: {metric: (point_est, ci_lower, ci_upper)}}

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for model_name, metrics in model_ci_results.items():
        row = {"Model": model_name}
        for metric_name, (point, lo, hi) in metrics.items():
            row[f"{metric_name}"] = f"{point:.4f} [{lo:.4f}, {hi:.4f}]"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


if __name__ == "__main__":
    # Quick smoke test
    y_true = [0, 1, 2, 0, 2, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 1, 2, 1, 0, 0, 1, 2]

    metrics = compute_metrics(y_true, y_pred)
    print("Accuracy:", metrics["accuracy"])
    print("Macro F1:", metrics["macro_f1"])
    print("Kappa:", metrics["cohens_kappa"])
    print("Per-class:", metrics["per_class"])

    df = classification_report_df(y_true, y_pred)
    print(df)

    fig = plot_confusion_matrix(y_true, y_pred, save_path=config.PLOTS_DIR / "cm_test.png")
    plt.close(fig)
    print("Confusion matrix saved.")

    # Bootstrap test
    rng = np.random.RandomState(42)
    y_true_big = rng.randint(0, 3, 500)
    y_pred_a = y_true_big.copy()
    y_pred_b = y_true_big.copy()
    noise_a = rng.choice(len(y_true_big), size=100, replace=False)
    noise_b = rng.choice(len(y_true_big), size=30, replace=False)
    y_pred_a[noise_a] = (y_pred_a[noise_a] + 1) % 3
    y_pred_b[noise_b] = (y_pred_b[noise_b] + 1) % 3

    ci = bootstrap_ci(y_true_big, y_pred_a, n_bootstrap=100)
    print("\nBootstrap CI for model A:")
    for m, (pt, lo, hi) in ci.items():
        print(f"  {m}: {pt:.4f} [{lo:.4f}, {hi:.4f}]")

    sig = bootstrap_paired_test(y_true_big, y_pred_a, y_pred_b, n_bootstrap=1000)
    print(f"\nPaired test: delta={sig['delta']:.4f}, p={sig['p_value']:.4f}")
