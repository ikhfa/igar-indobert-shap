"""
Evaluation metrics and reporting for sentiment classification models.

Computes: accuracy, macro F1/precision/recall, per-class metrics, Cohen's kappa.
Produces: confusion matrix plots, comparison tables.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
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
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

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
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig


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
