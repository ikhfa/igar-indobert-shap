"""
Comparative analysis, case studies, and domain keyword validation.

Combines model comparison results, misclassification analysis, and
SHAP-based case studies into publication-ready outputs.
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
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Domain keywords from the government app review domain
DOMAIN_KEYWORDS: List[str] = [
    "otp", "login", "server", "update", "error",
    "gagal", "lambat", "crash", "tidak bisa", "tidak jalan",
    "fitur", "data", "akses", "verifikasi", "registrasi",
    "loading", "koneksi", "notifikasi", "pembaruan", "layanan",
]


# ---------------------------------------------------------------------------
# Performance Comparison
# ---------------------------------------------------------------------------

def performance_comparison_table(
    baseline_results: Dict[str, Dict],
    indobert_results: Dict,
    save_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Build a styled performance comparison table and save as CSV.

    Parameters
    ----------
    baseline_results : Dict[str, Dict]
        Keys: model names (e.g. 'TF-IDF+RF'), values: compute_metrics() output.
    indobert_results : Dict
        compute_metrics() output for IndoBERT.
    save_path : str or Path, optional
        CSV save path. Defaults to config.METRICS_DIR / 'comparison.csv'.

    Returns
    -------
    pd.DataFrame
        Full comparison table.
    """
    from src.evaluate import compare_models_table

    all_results = dict(baseline_results)
    all_results["IndoBERT (fine-tuned)"] = indobert_results

    df = compare_models_table(all_results)

    _save = save_path or (config.METRICS_DIR / "comparison.csv")
    df.to_csv(str(_save))
    print(f"Comparison table saved to {_save}")
    return df


def plot_performance_comparison(
    results_dict: Dict[str, Dict],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing accuracy, macro F1, precision, recall, and kappa.

    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Keyed by model name.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    from src.evaluate import plot_performance_comparison as _plot

    _save = save_path or (config.PLOTS_DIR / "model_comparison.png")
    return _plot(results_dict, save_path=_save)


# ---------------------------------------------------------------------------
# Misclassification Analysis
# ---------------------------------------------------------------------------

def find_misclassified(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    device: Optional[torch.device] = None,
    text_col: str = "clean_text",
    label_col: str = config.LABEL_COL,
    batch_size: int = config.EVAL_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Find all misclassified samples in the test set.

    Parameters
    ----------
    model : IndoBERTClassifier
    tokenizer : PreTrainedTokenizerBase
    test_df : pd.DataFrame
        Test DataFrame with text_col and label_col.
    device : torch.device, optional
    text_col : str
        Column with preprocessed text.
    label_col : str
        True label column.
    batch_size : int

    Returns
    -------
    pd.DataFrame
        Rows where prediction != true label.
        Columns include: text, true_label, predicted_label, true_name, predicted_name,
                         confidence, all class probabilities.
    """
    _device = device or config.DEVICE
    model.eval()

    texts = test_df[text_col].tolist()
    true_labels = test_df[label_col].tolist()

    all_probs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting test set"):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            max_length=config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(_device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model.bert(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    all_probs_arr = np.vstack(all_probs)
    pred_labels = all_probs_arr.argmax(axis=1)
    confidences = all_probs_arr.max(axis=1)

    # Build result DataFrame
    result_df = test_df[[text_col, label_col]].copy()
    result_df["predicted_label"] = pred_labels
    result_df["true_name"] = result_df[label_col].map(config.LABEL_MAP)
    result_df["predicted_name"] = result_df["predicted_label"].map(config.LABEL_MAP)
    result_df["confidence"] = confidences
    for cls_idx, cls_name in config.LABEL_MAP.items():
        result_df[f"prob_{cls_name.lower()}"] = all_probs_arr[:, cls_idx]

    # Keep only misclassified
    misclassified = result_df[
        result_df[label_col] != result_df["predicted_label"]
    ].reset_index(drop=True)

    print(
        f"Misclassified: {len(misclassified)} / {len(test_df)} "
        f"({100 * len(misclassified) / len(test_df):.1f}%)"
    )

    # Save to CSV
    save_path = config.METRICS_DIR / "misclassified.csv"
    misclassified.to_csv(str(save_path), index=False)
    print(f"Misclassified examples saved to {save_path}")

    return misclassified


# ---------------------------------------------------------------------------
# Case Study Analysis
# ---------------------------------------------------------------------------

def case_study_analysis(
    misclassified_df: pd.DataFrame,
    explainer,
    n_cases: int = 10,
    text_col: str = "clean_text",
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate detailed SHAP explanations for misclassified reviews.

    For each case, computes SHAP values for both the true and predicted class
    to understand why the model was confused.

    Parameters
    ----------
    misclassified_df : pd.DataFrame
        Output of find_misclassified().
    explainer : SHAPExplainer
        Initialized SHAP explainer.
    n_cases : int
        Number of cases to analyze.
    text_col : str
        Column with review text.
    save_dir : Path, optional
        Directory to save waterfall plots.

    Returns
    -------
    pd.DataFrame
        Case study results with SHAP summaries.
    """
    _save_dir = save_dir or (config.PLOTS_DIR / "case_studies")
    _save_dir.mkdir(parents=True, exist_ok=True)

    cases = misclassified_df.head(n_cases)
    rows = []

    for idx, row in tqdm(cases.iterrows(), total=len(cases), desc="SHAP case studies"):
        text = str(row[text_col])
        true_name = str(row.get("true_name", ""))
        pred_name = str(row.get("predicted_name", ""))

        try:
            shap_data = explainer.explain_single(text)

            # Top 5 tokens for predicted class
            pred_tokens = sorted(
                shap_data.get(pred_name, {}).items(),
                key=lambda x: abs(x[1]), reverse=True
            )[:5]

            # Top 5 tokens for true class
            true_tokens = sorted(
                shap_data.get(true_name, {}).items(),
                key=lambda x: abs(x[1]), reverse=True
            )[:5]

            # Save waterfall plot
            fig = explainer.plot_waterfall(
                text, pred_name,
                save_path=_save_dir / f"case_{idx:04d}_{pred_name.lower()}.png",
            )
            plt.close(fig)

            rows.append({
                "case_id": idx,
                "text": text[:100],
                "true_class": true_name,
                "predicted_class": pred_name,
                "confidence": float(row.get("confidence", 0.0)),
                "top_tokens_predicted": str(pred_tokens),
                "top_tokens_true": str(true_tokens),
            })

        except Exception as e:
            print(f"  [Warning] Case {idx} failed: {e}")
            rows.append({
                "case_id": idx,
                "text": text[:100],
                "true_class": true_name,
                "predicted_class": pred_name,
                "confidence": float(row.get("confidence", 0.0)),
                "top_tokens_predicted": "error",
                "top_tokens_true": "error",
            })

    df_cases = pd.DataFrame(rows)
    save_path = config.METRICS_DIR / "case_studies.csv"
    df_cases.to_csv(str(save_path), index=False)
    print(f"\nCase study results saved to {save_path}")
    return df_cases


# ---------------------------------------------------------------------------
# Domain Keyword Validation
# ---------------------------------------------------------------------------

def domain_keyword_validation(
    token_importance_df: pd.DataFrame,
    domain_keywords: Optional[List[str]] = None,
    text_col: str = "token",
    value_col: str = "mean_shap",
) -> pd.DataFrame:
    """
    Check overlap between top SHAP tokens and domain-specific keywords.

    Domain keywords represent known important terms for government app reviews
    (e.g. 'otp', 'login', 'server', 'error', 'gagal').

    Parameters
    ----------
    token_importance_df : pd.DataFrame
        Output of SHAPExplainer.aggregate_token_importance().
    domain_keywords : List[str], optional
        Domain keyword list. Defaults to DOMAIN_KEYWORDS.
    text_col : str
        Column with token strings.
    value_col : str
        Column with SHAP values.

    Returns
    -------
    pd.DataFrame
        Rows: matched domain keywords with their SHAP rank and value.
    """
    _kws = domain_keywords or DOMAIN_KEYWORDS
    kw_set = set(kw.lower() for kw in _kws)

    matched = token_importance_df[
        token_importance_df[text_col].str.lower().isin(kw_set)
    ].copy()

    if matched.empty:
        print("No domain keywords found in top SHAP tokens.")
        # Return zero-row table with expected schema
        return pd.DataFrame(columns=[text_col, value_col, "class_name", "rank"])

    matched = matched.sort_values(value_col, key=lambda s: s.abs(), ascending=False)

    n_total = len(_kws)
    n_found = matched[text_col].nunique()
    coverage = n_found / n_total

    print(f"Domain keyword coverage: {n_found}/{n_total} ({coverage:.1%})")
    print("Matched keywords:")
    for _, row in matched.iterrows():
        print(f"  [{row.get('class_name', '?')}] {row[text_col]}: {row[value_col]:.4f} (rank {row.get('rank', '?')})")

    save_path = config.METRICS_DIR / "domain_keyword_validation.csv"
    matched.to_csv(str(save_path), index=False)
    print(f"Validation results saved to {save_path}")
    return matched


# ---------------------------------------------------------------------------
# Label Distribution Plot
# ---------------------------------------------------------------------------

def plot_class_distribution(
    df: pd.DataFrame,
    label_col: str = config.LABEL_COL,
    app_col: Optional[str] = config.APP_COL,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot class distribution overall and per app (if app_col provided).

    Parameters
    ----------
    df : pd.DataFrame
    label_col : str
    app_col : str, optional
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_plots = 2 if (app_col and app_col in df.columns) else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Overall distribution
    counts = df[label_col].map(config.LABEL_MAP).value_counts()
    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    axes[0].bar(config.LABEL_NAMES, [counts.get(n, 0) for n in config.LABEL_NAMES], color=colors)
    axes[0].set_title("Overall Class Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate([counts.get(n, 0) for n in config.LABEL_NAMES]):
        axes[0].text(i, v + 5, str(v), ha="center")

    # Per-app distribution
    if n_plots == 2:
        pivot = df.groupby([app_col, label_col]).size().unstack(fill_value=0)
        pivot.columns = [config.LABEL_MAP.get(c, str(c)) for c in pivot.columns]
        pivot[config.LABEL_NAMES].plot(
            kind="bar", ax=axes[1],
            color=colors, stacked=False,
        )
        axes[1].set_title("Class Distribution per App")
        axes[1].set_xlabel("Application")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].legend(title="Sentiment")

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    from src.preprocessing import load_dataset, preprocess_dataframe, remove_duplicates, split_data

    config.set_all_seeds()

    df = load_dataset()
    df = preprocess_dataframe(df)
    df = remove_duplicates(df)

    fig = plot_class_distribution(df, save_path=config.PLOTS_DIR / "class_dist.png")
    plt.close(fig)
    print("Class distribution plot saved.")

    # Mock comparison table
    mock_results = {
        "TF-IDF+RF": {"accuracy": 0.82, "macro_f1": 0.79, "macro_precision": 0.81,
                      "macro_recall": 0.78, "cohens_kappa": 0.70, "per_class": {}},
        "TF-IDF+SVC": {"accuracy": 0.85, "macro_f1": 0.83, "macro_precision": 0.84,
                       "macro_recall": 0.82, "cohens_kappa": 0.75, "per_class": {}},
        "IndoBERT (fine-tuned)": {"accuracy": 0.91, "macro_f1": 0.90, "macro_precision": 0.91,
                                  "macro_recall": 0.89, "cohens_kappa": 0.86, "per_class": {}},
    }
    fig = plot_performance_comparison(mock_results)
    plt.close(fig)
    print("Comparison plot saved.")
