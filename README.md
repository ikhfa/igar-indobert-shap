# Explainable Sentiment Classification of Indonesian Government Application Reviews Using Fine-Tuned IndoBERT and SHAP-Based Interpretation

Replication and extension of Isnan & Pardamean (2026) — comparing TF-IDF baselines with fine-tuned IndoBERT, explained via SHAP token attributions.

## Overview

This project:
1. Loads and preprocesses Indonesian app review data (617K records from Mendeley)
2. Replicates TF-IDF + RF and TF-IDF + LinearSVC baselines
3. Fine-tunes `indobenchmark/indobert-base-p1` for 3-class sentiment (Negative/Neutral/Positive)
4. Applies SHAP (SHapley Additive exPlanations) for token-level attribution
5. Analyzes misclassifications and validates against domain keywords

## Project Structure

```
├── pyproject.toml             # Project metadata & dependencies (uv/pip)
├── run_pipeline.py            # Full pipeline runner (all steps in one command)
├── config.py                  # All hyperparameters and settings
├── requirements.txt           # Pip-compatible dependency list
├── data/
│   └── README.md              # Dataset acquisition instructions
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Loading, cleaning, slang normalization, splitting
│   ├── baseline.py            # TF-IDF + RF + LinearSVC replication
│   ├── dataset.py             # IndoBERTDataset (PyTorch)
│   ├── train.py               # Fine-tuning loop with early stopping
│   ├── evaluate.py            # Metrics, confusion matrix, comparison table
│   ├── explain.py             # SHAP token attribution
│   └── analysis.py            # Comparison plots, case studies, keyword validation
├── notebooks/
│   └── full_pipeline.ipynb    # End-to-end walkthrough
└── output/                    # Generated (gitignored)
    ├── models/
    ├── metrics/
    ├── plots/
    └── logs/
```

## Setup

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or use pip.

Python 3.9+ required. CUDA optional but significantly speeds up IndoBERT training.

### Install dependencies

Using **uv** (recommended):

```bash
uv sync
```

Using pip:

```bash
pip install -r requirements.txt
```

### Dataset

See `data/README.md` for instructions. Place `Rating_labeled.csv` in the `data/` directory.

## Running the Pipeline

### Full pipeline (all steps)

Run the entire pipeline end-to-end with a single command:

```bash
uv run python run_pipeline.py
```

This executes all 6 steps sequentially: preprocessing, baselines, IndoBERT fine-tuning, evaluation, SHAP analysis, and case studies.

### Run specific steps

```bash
uv run python run_pipeline.py --step 1      # Only preprocessing
uv run python run_pipeline.py --step 3      # Only IndoBERT training
uv run python run_pipeline.py --step 1-3    # Steps 1 through 3
uv run python run_pipeline.py --step 4-6    # Evaluation, SHAP, and analysis
```

| Step | Description |
| ---- | ----------- |
| 1 | Preprocessing — load, clean, normalize slang, deduplicate, split 80/10/10 |
| 2 | Baselines — TF-IDF + RandomForest and TF-IDF + LinearSVC |
| 3 | IndoBERT — fine-tune `indobenchmark/indobert-base-p1` with early stopping |
| 4 | Evaluation — test metrics, confusion matrix, model comparison |
| 5 | SHAP — token attributions, top-20 tokens per class |
| 6 | Analysis — misclassifications, case studies, domain keyword validation |

> Steps 4-6 depend on a trained model. If run independently, they will load the saved checkpoint from `output/models/indobert_best.pt`.

### Run individual modules

Each module can also be run standalone:

```bash
uv run python src/preprocessing.py
uv run python src/baseline.py
uv run python src/train.py
uv run python src/evaluate.py
uv run python src/explain.py
uv run python src/analysis.py
```

### Interactive notebook

```bash
uv run jupyter notebook notebooks/full_pipeline.ipynb
```

## Configuration

All settings in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `indobenchmark/indobert-base-p1` | Base IndoBERT |
| `LEARNING_RATE` | `2e-5` | AdamW peak LR |
| `BATCH_SIZE` | `32` | Training batch size |
| `NUM_EPOCHS` | `5` | Max training epochs |
| `MAX_LEN` | `128` | Token sequence length |
| `TFIDF_MAX_FEATURES` | `5000` | TF-IDF vocabulary size |
| `RANDOM_SEED` | `42` | Global random seed |

## Expected Outputs

| Path | Description |
|------|-------------|
| `output/models/indobert_best.pt` | Best IndoBERT checkpoint |
| `output/logs/training_metrics.csv` | Per-epoch train/val metrics |
| `output/metrics/comparison.csv` | Model comparison table |
| `output/metrics/misclassified.csv` | Misclassified test samples |
| `output/metrics/domain_keyword_validation.csv` | SHAP-keyword overlap |
| `output/plots/` | All generated figures |

## Reference

Isnan, Mahmud; Pardamean, Bens  (2025), “IGAR: Indonesian Government App Review Dataset”, Mendeley Data, V3, doi: 10.17632/7zryc6k76z.3
