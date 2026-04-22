"""
Comparative analysis, case studies, and domain keyword validation.

Combines model comparison results, misclassification analysis, and
SHAP-based case studies into publication-ready outputs.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

tqdm = config.get_tqdm()

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
    print(f"\nCase study analysis complete ({len(df_cases)} cases).")
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
    colors = config.LABEL_COLORS
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
        fig.savefig(str(save_path), dpi=config.PLOT_DPI, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Per-App Evaluation
# ---------------------------------------------------------------------------

def per_app_evaluation(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    device: Optional[torch.device] = None,
    text_col: str = "clean_text",
    label_col: str = config.LABEL_COL,
    app_col: str = config.APP_COL,
    batch_size: int = config.EVAL_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Compute per-app metrics breakdown for IndoBERT on the test set.

    Parameters
    ----------
    model : IndoBERTClassifier
    tokenizer : PreTrainedTokenizerBase
    test_df : pd.DataFrame
        Must contain text_col, label_col, and app_col.
    device : torch.device, optional
    text_col : str
    label_col : str
    app_col : str
    batch_size : int

    Returns
    -------
    pd.DataFrame
        One row per app, with accuracy, macro_f1, per-class F1, and count.
    """
    from src.evaluate import compute_metrics

    _device = device or config.DEVICE
    model.eval()

    apps = test_df[app_col].unique()
    rows = []

    for app_name in sorted(apps):
        app_mask = test_df[app_col] == app_name
        app_df = test_df[app_mask].reset_index(drop=True)
        if len(app_df) < 10:
            print(f"  Skipping {app_name}: only {len(app_df)} samples.")
            continue

        texts = app_df[text_col].tolist()
        true_labels = app_df[label_col].tolist()

        all_preds = []
        for i in range(0, len(texts), batch_size):
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
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)

        metrics = compute_metrics(true_labels, all_preds)

        row = {
            "app": app_name,
            "n_samples": len(app_df),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "cohens_kappa": metrics["cohens_kappa"],
        }
        for cls_name in config.LABEL_NAMES:
            pc = metrics.get("per_class", {}).get(cls_name, {})
            row[f"f1_{cls_name.lower()}"] = pc.get("f1", float("nan"))

        rows.append(row)
        print(
            f"  {app_name} ({len(app_df):,} samples): "
            f"Acc={metrics['accuracy']:.4f} | Macro-F1={metrics['macro_f1']:.4f} | "
            f"Kappa={metrics['cohens_kappa']:.4f}"
        )

    result_df = pd.DataFrame(rows)
    print(f"\nPer-app evaluation complete: {len(result_df)} apps.")
    return result_df


def per_app_baseline_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = config.LABEL_COL,
    app_col: str = config.APP_COL,
) -> pd.DataFrame:
    """
    Compute per-app metrics for TF-IDF baselines.

    Fits the TF-IDF vectorizer on the full training set, then evaluates
    per-app subsets of the test set.

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    text_col : str
    label_col : str
    app_col : str

    Returns
    -------
    pd.DataFrame
        One row per (app, model) pair.
    """
    from src.baseline import build_pipeline
    from src.evaluate import compute_metrics

    X_train = train_df[text_col].tolist()
    y_train = train_df[label_col].tolist()

    rows = []

    for model_type, model_name in [("rf", "TF-IDF+RF"), ("svc", "TF-IDF+SVC")]:
        pipeline = build_pipeline(model_type)
        pipeline.fit(X_train, y_train)

        for app_name in sorted(test_df[app_col].unique()):
            app_mask = test_df[app_col] == app_name
            app_df = test_df[app_mask].reset_index(drop=True)
            if len(app_df) < 10:
                continue

            X_test = app_df[text_col].tolist()
            y_test = app_df[label_col].tolist()
            y_pred = pipeline.predict(X_test)

            metrics = compute_metrics(y_test, list(y_pred))

            row = {
                "app": app_name,
                "model": model_name,
                "n_samples": len(app_df),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "cohens_kappa": metrics["cohens_kappa"],
            }
            for cls_name in config.LABEL_NAMES:
                pc = metrics.get("per_class", {}).get(cls_name, {})
                row[f"f1_{cls_name.lower()}"] = pc.get("f1", float("nan"))

            rows.append(row)
            print(
                f"  {app_name} / {model_name}: "
                f"Acc={metrics['accuracy']:.4f} | Macro-F1={metrics['macro_f1']:.4f}"
            )

    result_df = pd.DataFrame(rows)
    print(f"\nPer-app baseline evaluation complete: {len(result_df)} entries.")
    return result_df


def combined_per_app_table(
    baseline_per_app_df: pd.DataFrame,
    indobert_per_app_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Merge baseline and IndoBERT per-app results into a single comparison table.

    Parameters
    ----------
    baseline_per_app_df : pd.DataFrame
        Output of per_app_baseline_evaluation().
    indobert_per_app_df : pd.DataFrame
        Output of per_app_evaluation().
    save_path : str or Path, optional

    Returns
    -------
    pd.DataFrame
    """
    indobert_rows = indobert_per_app_df.copy()
    indobert_rows["model"] = "IndoBERT (fine-tuned)"

    baseline_rows = baseline_per_app_df[
        [c for c in indobert_rows.columns if c in baseline_per_app_df.columns]
    ].copy()

    combined = pd.concat([baseline_rows, indobert_rows], ignore_index=True)

    _save = save_path or (config.METRICS_DIR / "per_app_comparison.csv")
    combined.to_csv(str(_save), index=False)
    print(f"Per-app comparison table saved to {_save}")
    return combined


def plot_per_app_comparison(
    combined_df: pd.DataFrame,
    metric: str = "macro_f1",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing a metric across apps and models.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Output of combined_per_app_table().
    metric : str
        Column name to plot (e.g. 'macro_f1', 'accuracy').
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    apps = sorted(combined_df["app"].unique())
    models = combined_df["model"].unique()
    n_apps = len(apps)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(max(10, n_apps * 2), 6))
    x = np.arange(n_apps)
    width = 0.8 / n_models
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, model_name in enumerate(models):
        model_data = combined_df[combined_df["model"] == model_name]
        values = []
        for app in apps:
            app_row = model_data[model_data["app"] == app]
            values.append(app_row[metric].values[0] if len(app_row) > 0 else 0.0)

        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width * 0.9, label=model_name, color=colors[i])
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(apps, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Per-App {metric.replace('_', ' ').title()} Comparison", fontsize=13)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=config.PLOT_DPI, bbox_inches="tight")
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
