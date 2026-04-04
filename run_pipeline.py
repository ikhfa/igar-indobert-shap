"""
Full pipeline runner for IndoBERT SHAP Sentiment Classification.

Usage:
    uv run python run_pipeline.py              # Run all steps
    uv run python run_pipeline.py --step 1     # Run only step 1 (preprocessing)
    uv run python run_pipeline.py --step 1-3   # Run steps 1 through 3
"""

import argparse
import sys
import time

import config

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_1_preprocessing():
    """Load, clean, normalize, deduplicate, and split data."""
    print("=" * 70)
    print("STEP 1: Preprocessing")
    print("=" * 70)
    from src.preprocessing import load_dataset, preprocess_dataframe, remove_duplicates, split_data

    df = load_dataset()
    print(f"Loaded {len(df)} records.")
    print(f"Label distribution:\n{df[config.LABEL_COL].value_counts()}\n")

    df = preprocess_dataframe(df)
    df = remove_duplicates(df)
    train_df, val_df, test_df = split_data(df)

    print(f"\nSplit sizes — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def step_2_baselines(train_df, test_df):
    """Train TF-IDF + RF and TF-IDF + LinearSVC baselines."""
    print("\n" + "=" * 70)
    print("STEP 2: Baseline Models (TF-IDF + RF / LinearSVC)")
    print("=" * 70)
    from src.baseline import run_all_baselines

    results = run_all_baselines(train_df, test_df)
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Macro F1:   {metrics['macro_f1']:.4f}")
        print(f"  Kappa:      {metrics['cohens_kappa']:.4f}")
    return results


def step_3_train_indobert(train_df, val_df, test_df):
    """Fine-tune IndoBERT and build dataloaders."""
    print("\n" + "=" * 70)
    print("STEP 3: IndoBERT Fine-tuning")
    print("=" * 70)
    from src.train import load_tokenizer, fine_tune_indobert
    from src.dataset import build_dataloaders

    tokenizer = load_tokenizer()
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df["clean_text"].tolist(), train_df[config.LABEL_COL].tolist(),
        val_df["clean_text"].tolist(), val_df[config.LABEL_COL].tolist(),
        test_df["clean_text"].tolist(), test_df[config.LABEL_COL].tolist(),
        tokenizer,
    )

    model, history = fine_tune_indobert(train_loader, val_loader)
    print(f"\nTraining complete. Best Val Macro F1: {max(history['val_macro_f1']):.4f}")
    return model, tokenizer, test_loader, history


def step_4_evaluate(model, tokenizer, test_loader, baseline_results):
    """Evaluate IndoBERT on test set and compare with baselines."""
    print("\n" + "=" * 70)
    print("STEP 4: Evaluation & Comparison")
    print("=" * 70)
    import matplotlib.pyplot as plt
    from src.train import eval_epoch
    from src.evaluate import compute_metrics, plot_confusion_matrix, classification_report_df
    from src.analysis import performance_comparison_table, plot_performance_comparison

    _, y_pred, y_true = eval_epoch(model, test_loader, config.DEVICE)
    indobert_metrics = compute_metrics(y_true, y_pred)

    print("\nIndoBERT Test Results:")
    print(f"  Accuracy:   {indobert_metrics['accuracy']:.4f}")
    print(f"  Macro F1:   {indobert_metrics['macro_f1']:.4f}")
    print(f"  Kappa:      {indobert_metrics['cohens_kappa']:.4f}")

    report = classification_report_df(y_true, y_pred)
    print(f"\n{report}\n")

    # Confusion matrix
    fig = plot_confusion_matrix(y_true, y_pred, save_path=config.PLOTS_DIR / "cm_indobert.png")
    plt.close(fig)
    print("Confusion matrix saved to output/plots/cm_indobert.png")

    # Model comparison
    all_results = {**baseline_results, "IndoBERT (fine-tuned)": indobert_metrics}
    comparison_df = performance_comparison_table(baseline_results, indobert_metrics,
                                                  save_path=config.METRICS_DIR / "comparison.csv")
    print(f"\n{comparison_df}\n")

    fig = plot_performance_comparison(all_results, save_path=config.PLOTS_DIR / "model_comparison.png")
    plt.close(fig)
    print("Comparison plot saved to output/plots/model_comparison.png")

    return indobert_metrics, y_pred, y_true


def step_5_shap(model, tokenizer, test_df):
    """Compute SHAP token attributions."""
    print("\n" + "=" * 70)
    print("STEP 5: SHAP Analysis")
    print("=" * 70)
    import matplotlib.pyplot as plt
    from src.explain import SHAPExplainer

    explainer = SHAPExplainer(model, tokenizer, config.DEVICE)

    # Use a subset for SHAP (computationally expensive)
    shap_texts = test_df["clean_text"].tolist()[:50]
    print(f"Computing SHAP for {len(shap_texts)} test samples...")

    shap_results = explainer.explain_batch(shap_texts)
    token_df = explainer.aggregate_token_importance(shap_results)

    for cls_name in config.LABEL_NAMES:
        cls_df = token_df[token_df["class_name"] == cls_name]
        if not cls_df.empty:
            fig = explainer.plot_shap_summary(
                cls_df, cls_name,
                save_path=config.PLOTS_DIR / f"shap_top20_{cls_name.lower()}.png",
            )
            plt.close(fig)

    print("SHAP summary plots saved to output/plots/")
    print(f"\nTop aggregate tokens:\n{token_df.head(10).to_string(index=False)}")
    return explainer, shap_results, token_df


def step_6_analysis(model, tokenizer, test_df, explainer, shap_results, token_df):
    """Misclassification case studies and domain keyword validation."""
    print("\n" + "=" * 70)
    print("STEP 6: Analysis & Case Studies")
    print("=" * 70)
    import matplotlib.pyplot as plt
    from src.analysis import (
        find_misclassified,
        case_study_analysis,
        domain_keyword_validation,
        plot_class_distribution,
    )

    # Class distribution
    from src.preprocessing import load_dataset, preprocess_dataframe, remove_duplicates
    df_full = load_dataset()
    df_full = preprocess_dataframe(df_full)
    df_full = remove_duplicates(df_full)
    fig = plot_class_distribution(df_full, save_path=config.PLOTS_DIR / "class_distribution.png")
    plt.close(fig)
    print("Class distribution plot saved.")

    # Misclassifications
    misclassified = find_misclassified(model, tokenizer, test_df)
    print(f"Found {len(misclassified)} misclassified samples.")
    misclassified.to_csv(config.METRICS_DIR / "misclassified.csv", index=False)

    # Case studies
    if len(misclassified) > 0:
        cases = case_study_analysis(
            misclassified, explainer, n_cases=min(5, len(misclassified)),
            save_dir=config.PLOTS_DIR / "case_studies",
        )
        cases.to_csv(config.METRICS_DIR / "case_studies.csv", index=False)
        print(f"Case studies saved ({len(cases)} cases).")

    # Domain keyword validation
    kw_df = domain_keyword_validation(token_df)
    kw_df.to_csv(config.METRICS_DIR / "domain_keyword_validation.csv", index=False)
    print(f"Domain keyword validation: {len(kw_df)} keywords matched in SHAP rankings.")

    print("\nAll outputs saved to output/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_step_range(step_arg: str):
    """Parse step argument like '1', '2-5', or None (all)."""
    if step_arg is None:
        return 1, 6
    if "-" in step_arg:
        start, end = step_arg.split("-", 1)
        return int(start), int(end)
    n = int(step_arg)
    return n, n


def main():
    parser = argparse.ArgumentParser(
        description="Run the IndoBERT SHAP sentiment classification pipeline.",
    )
    parser.add_argument(
        "--step", type=str, default=None,
        help="Step(s) to run: '1', '3-5', or omit for all (1-6).",
    )
    args = parser.parse_args()

    start_step, end_step = parse_step_range(args.step)
    print(f"Running steps {start_step}–{end_step}")
    print(f"Device: {config.DEVICE}")
    print(f"Sample mode: {config.USE_SAMPLE}\n")

    config.set_all_seeds()
    t0 = time.time()

    # Shared state across steps
    train_df = val_df = test_df = None
    baseline_results = None
    model = tokenizer = test_loader = history = None
    indobert_metrics = None
    explainer = shap_results = token_df = None

    # Step 1
    if start_step <= 1 <= end_step:
        train_df, val_df, test_df = step_1_preprocessing()

    # Step 2
    if start_step <= 2 <= end_step:
        if train_df is None:
            train_df, val_df, test_df = step_1_preprocessing()
        baseline_results = step_2_baselines(train_df, test_df)

    # Step 3
    if start_step <= 3 <= end_step:
        if train_df is None:
            train_df, val_df, test_df = step_1_preprocessing()
        model, tokenizer, test_loader, history = step_3_train_indobert(train_df, val_df, test_df)

    # Step 4
    if start_step <= 4 <= end_step:
        if model is None:
            # Load trained model
            from src.train import load_tokenizer, load_trained_model
            from src.dataset import build_dataloaders
            if train_df is None:
                train_df, val_df, test_df = step_1_preprocessing()
            tokenizer = load_tokenizer()
            model = load_trained_model()
            _, _, test_loader = build_dataloaders(
                train_df["clean_text"].tolist(), train_df[config.LABEL_COL].tolist(),
                val_df["clean_text"].tolist(), val_df[config.LABEL_COL].tolist(),
                test_df["clean_text"].tolist(), test_df[config.LABEL_COL].tolist(),
                tokenizer,
            )
        if baseline_results is None:
            baseline_results = step_2_baselines(train_df, test_df)
        indobert_metrics, _, _ = step_4_evaluate(model, tokenizer, test_loader, baseline_results)

    # Step 5
    if start_step <= 5 <= end_step:
        if model is None:
            from src.train import load_tokenizer, load_trained_model
            if train_df is None:
                train_df, val_df, test_df = step_1_preprocessing()
            tokenizer = load_tokenizer()
            model = load_trained_model()
        explainer, shap_results, token_df = step_5_shap(model, tokenizer, test_df)

    # Step 6
    if start_step <= 6 <= end_step:
        if model is None:
            from src.train import load_tokenizer, load_trained_model
            if train_df is None:
                train_df, val_df, test_df = step_1_preprocessing()
            tokenizer = load_tokenizer()
            model = load_trained_model()
        if explainer is None:
            explainer, shap_results, token_df = step_5_shap(model, tokenizer, test_df)
        step_6_analysis(model, tokenizer, test_df, explainer, shap_results, token_df)

    elapsed = time.time() - t0
    print(f"\nPipeline finished in {elapsed / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
