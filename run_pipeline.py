"""
Full pipeline runner for IndoBERT SHAP Sentiment Classification.

Usage:
    uv run python run_pipeline.py              # Run all steps
    uv run python run_pipeline.py --step 1     # Run only step 1 (preprocessing)
    uv run python run_pipeline.py --step 1-3   # Run steps 1 through 3
"""

import argparse
import json
import sys
import time
from datetime import datetime

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


def step_4_evaluate(model, tokenizer, test_loader, baseline_results, train_df, test_df):
    """Evaluate IndoBERT on test set, compare with baselines, run significance tests."""
    print("\n" + "=" * 70)
    print("STEP 4: Evaluation & Comparison")
    print("=" * 70)
    import matplotlib.pyplot as plt
    from src.train import eval_epoch
    from src.evaluate import (
        compute_metrics, plot_confusion_matrix, classification_report_df,
        bootstrap_ci, bootstrap_paired_test, bootstrap_ci_table,
        significance_comparison_table,
    )
    from src.analysis import (
        performance_comparison_table, plot_performance_comparison,
        per_app_evaluation, per_app_baseline_evaluation,
        combined_per_app_table, plot_per_app_comparison,
    )

    _, y_pred, y_true = eval_epoch(model, test_loader, config.DEVICE)
    indobert_metrics = compute_metrics(y_true, y_pred)

    print("\nIndoBERT Test Results:")
    print(f"  Accuracy:   {indobert_metrics['accuracy']:.4f}")
    print(f"  Macro F1:   {indobert_metrics['macro_f1']:.4f}")
    print(f"  Kappa:      {indobert_metrics['cohens_kappa']:.4f}")

    report = classification_report_df(y_true, y_pred)
    print(f"\n{report}\n")

    fig = plot_confusion_matrix(y_true, y_pred, save_path=config.PLOTS_DIR / "cm_indobert.png")
    plt.close(fig)
    print("Confusion matrix saved to output/plots/cm_indobert.png")

    all_results = {**baseline_results, "IndoBERT (fine-tuned)": indobert_metrics}
    comparison_df = performance_comparison_table(baseline_results, indobert_metrics,
                                                  save_path=config.METRICS_DIR / "comparison.csv")
    print(f"\n{comparison_df}\n")

    fig = plot_performance_comparison(all_results, save_path=config.PLOTS_DIR / "model_comparison.png")
    plt.close(fig)
    print("Comparison plot saved to output/plots/model_comparison.png")

    # --- Bootstrap confidence intervals ---
    print("\n--- Bootstrap 95% Confidence Intervals ---")
    indobert_ci = bootstrap_ci(y_true, y_pred, n_bootstrap=1000, seed=config.RANDOM_SEED)
    baseline_preds = {}
    for model_name in baseline_results:
        if "RF" in model_name:
            from src.baseline import build_pipeline
            pipe = build_pipeline("rf")
            pipe.fit(train_df["clean_text"].tolist(), train_df[config.LABEL_COL].tolist())
            baseline_preds[model_name] = pipe.predict(test_df["clean_text"].tolist()).tolist()
        elif "SVC" in model_name:
            from src.baseline import build_pipeline
            pipe = build_pipeline("svc")
            pipe.fit(train_df["clean_text"].tolist(), train_df[config.LABEL_COL].tolist())
            baseline_preds[model_name] = pipe.predict(test_df["clean_text"].tolist()).tolist()

    all_cis = {}
    for bname, bpreds in baseline_preds.items():
        all_cis[bname] = bootstrap_ci(y_true, bpreds, n_bootstrap=1000, seed=config.RANDOM_SEED)
    all_cis["IndoBERT (fine-tuned)"] = indobert_ci

    ci_df = bootstrap_ci_table(all_cis)
    ci_df.to_csv(config.METRICS_DIR / "bootstrap_ci.csv")
    print(ci_df.to_string())

    # --- Pairwise significance tests ---
    print("\n--- Pairwise Significance Tests (vs TF-IDF+SVC baseline) ---")
    all_model_preds = dict(baseline_preds)
    all_model_preds["IndoBERT (fine-tuned)"] = y_pred

    baseline_key = "TF-IDF+SVC" if "TF-IDF+SVC" in all_model_preds else list(all_model_preds.keys())[0]
    sig_df = significance_comparison_table(
        y_true, all_model_preds, baseline_name=baseline_key,
        n_bootstrap=10000, seed=config.RANDOM_SEED,
    )
    sig_df.to_csv(config.METRICS_DIR / "significance_tests.csv", index=False)
    print(sig_df.to_string(index=False))

    # --- Per-app evaluation ---
    print("\n--- Per-App Evaluation ---")
    indobert_per_app = per_app_evaluation(model, tokenizer, test_df)
    indobert_per_app.to_csv(config.METRICS_DIR / "per_app_indobert.csv", index=False)

    baseline_per_app = per_app_baseline_evaluation(train_df, test_df)
    combined_per_app = combined_per_app_table(
        baseline_per_app, indobert_per_app,
        save_path=config.METRICS_DIR / "per_app_comparison.csv",
    )

    fig = plot_per_app_comparison(combined_per_app, metric="macro_f1",
                                  save_path=config.PLOTS_DIR / "per_app_macro_f1.png")
    plt.close(fig)
    fig = plot_per_app_comparison(combined_per_app, metric="accuracy",
                                  save_path=config.PLOTS_DIR / "per_app_accuracy.png")
    plt.close(fig)
    print("Per-app comparison plots saved.")

    return indobert_metrics, y_pred, y_true


def step_5_shap(model, tokenizer, test_df):
    """Compute SHAP token attributions with stratified sampling."""
    print("\n" + "=" * 70)
    print("STEP 5: SHAP Analysis")
    print("=" * 70)
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedShuffleSplit
    from src.explain import SHAPExplainer

    explainer = SHAPExplainer(model, tokenizer, config.DEVICE)

    n_shap = min(config.SHAP_NUM_SAMPLES, len(test_df))
    if n_shap < len(test_df):
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=n_shap, random_state=config.RANDOM_SEED,
        )
        _, shap_indices = next(splitter.split(test_df, test_df[config.LABEL_COL]))
        shap_texts = test_df.iloc[shap_indices]["clean_text"].tolist()
    else:
        shap_texts = test_df["clean_text"].tolist()
    print(f"Computing SHAP for {len(shap_texts)} test samples (stratified)...")

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


def step_6_analysis(model, tokenizer, test_df, explainer, shap_results, token_df,
                    full_df=None):
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

    if full_df is not None:
        fig = plot_class_distribution(full_df, save_path=config.PLOTS_DIR / "class_distribution.png")
        plt.close(fig)
        print("Class distribution plot saved.")

    misclassified = find_misclassified(model, tokenizer, test_df)
    print(f"Found {len(misclassified)} misclassified samples.")
    misclassified.to_csv(config.METRICS_DIR / "misclassified.csv", index=False)

    if len(misclassified) > 0:
        cases = case_study_analysis(
            misclassified, explainer, n_cases=min(config.NUM_CASE_STUDIES, len(misclassified)),
            save_dir=config.PLOTS_DIR / "case_studies",
        )
        cases.to_csv(config.METRICS_DIR / "case_studies.csv", index=False)
        print(f"Case studies saved ({len(cases)} cases).")

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
    print(f"Device: {config.DEVICE}\n")

    config.set_all_seeds()
    t0 = time.time()

    run_meta = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "config": {k: str(v) for k, v in vars(config).items() if k.isupper()},
    }
    try:
        import git
        run_meta["git_commit"] = git.Repo(config.PROJECT_ROOT).head.commit.hexsha
    except Exception:
        run_meta["git_commit"] = "unknown"
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (config.LOGS_DIR / "run_config.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )

    # Shared state across steps
    train_df = val_df = test_df = None
    full_df = None
    baseline_results = None
    model = tokenizer = test_loader = history = None
    indobert_metrics = None
    explainer = shap_results = token_df = None

    # Step 1
    if start_step <= 1 <= end_step:
        train_df, val_df, test_df = step_1_preprocessing()
        full_df = None

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
        indobert_metrics, _, _ = step_4_evaluate(
            model, tokenizer, test_loader, baseline_results, train_df, test_df,
        )

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
        step_6_analysis(model, tokenizer, test_df, explainer, shap_results, token_df,
                        full_df=full_df)

    elapsed = time.time() - t0
    print(f"\nPipeline finished in {elapsed / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
