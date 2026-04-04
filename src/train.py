"""
IndoBERT fine-tuning for 3-class Indonesian sentiment classification.

Architecture: AutoModelForSequenceClassification on indobenchmark/indobert-base-p1
Optimizer: AdamW with linear warmup and weight decay
Training: Early stopping on validation macro-F1
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class IndoBERTClassifier(nn.Module):
    """
    IndoBERT fine-tuned for sequence classification.

    Wraps AutoModelForSequenceClassification with a 3-class linear head
    on the [CLS] token representation.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    num_labels : int
        Number of output classes.
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        num_labels: int = config.NUM_LABELS,
    ) -> None:
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            Shape (B, seq_len).
        attention_mask : torch.Tensor
            Shape (B, seq_len).
        token_type_ids : torch.Tensor, optional
            Shape (B, seq_len).
        labels : torch.Tensor, optional
            Shape (B,). If provided, loss is computed.

        Returns
        -------
        dict
            Keys: 'logits' (always), 'loss' (when labels provided).
        """
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        if labels is not None:
            kwargs["labels"] = labels

        outputs = self.bert(**kwargs)
        result = {"logits": outputs.logits}
        if labels is not None:
            result["loss"] = outputs.loss
        return result


# ---------------------------------------------------------------------------
# Epoch Functions
# ---------------------------------------------------------------------------

def train_epoch(
    model: IndoBERTClassifier,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Run one full training epoch.

    Parameters
    ----------
    model : IndoBERTClassifier
    loader : DataLoader
        Training DataLoader.
    optimizer : AdamW
    scheduler : LR scheduler
    device : torch.device

    Returns
    -------
    Tuple[float, float]
        (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False, mininterval=30)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids, labels)
        loss = outputs["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader), correct / total


def eval_epoch(
    model: IndoBERTClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, List[int], List[int]]:
    """
    Run one evaluation pass (val or test).

    Parameters
    ----------
    model : IndoBERTClassifier
    loader : DataLoader
    device : torch.device

    Returns
    -------
    Tuple[float, List[int], List[int]]
        (average_loss, all_predictions, all_true_labels)
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Eval ", leave=False, mininterval=30):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids, labels)
            total_loss += outputs["loss"].item()

            preds = outputs["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / len(loader), all_preds, all_labels


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def fine_tune_indobert(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str = config.MODEL_NAME,
    num_epochs: int = config.NUM_EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    warmup_ratio: float = config.WARMUP_RATIO,
    early_stopping_patience: int = config.EARLY_STOPPING_PATIENCE,
    device: Optional[torch.device] = None,
    save_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> Tuple[IndoBERTClassifier, Dict]:
    """
    Full fine-tuning loop for IndoBERT with early stopping.

    Training strategy:
    - AdamW optimizer with linear warmup (10% of total steps) and cosine decay
    - Gradient clipping at norm=1.0
    - Early stopping on validation macro-F1 (patience=2 by default)
    - Saves best model checkpoint to save_path

    Parameters
    ----------
    train_loader : DataLoader
        Training data.
    val_loader : DataLoader
        Validation data for early stopping.
    model_name : str
        HuggingFace model identifier.
    num_epochs : int
        Maximum training epochs.
    learning_rate : float
        Peak learning rate for AdamW.
    weight_decay : float
        L2 regularization.
    warmup_ratio : float
        Fraction of total steps for linear warmup.
    early_stopping_patience : int
        Number of epochs without improvement before stopping.
    device : torch.device, optional
        Defaults to config.DEVICE.
    save_path : Path, optional
        Where to save best model weights. Defaults to config.BEST_MODEL_PATH.
    log_path : Path, optional
        CSV file for per-epoch metrics. Defaults to config.METRICS_LOG_PATH.

    Returns
    -------
    Tuple[IndoBERTClassifier, Dict]
        (trained_model, training_history)
    """
    from src.evaluate import compute_metrics

    _device = device or config.DEVICE
    _save_path = save_path or config.BEST_MODEL_PATH
    _log_path = log_path or config.METRICS_LOG_PATH

    print(f"Device: {_device}")
    print(f"Loading model: {model_name}")

    model = IndoBERTClassifier(model_name, config.NUM_LABELS)
    model.to(_device)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-8,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Metrics log
    log_rows = []
    best_macro_f1 = -1.0
    patience_counter = 0
    history: Dict = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_accuracy": []}

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, _device)
        val_loss, val_preds, val_labels = eval_epoch(model, val_loader, _device)
        val_metrics = compute_metrics(val_labels, val_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\n"
            f"  Val   Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f}"
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })

        # Save best model
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            torch.save(model.state_dict(), str(_save_path))
            print(f"  [OK] New best model saved (macro F1: {best_macro_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Write metrics log
    if log_rows:
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"\nTraining log saved to {_log_path}")

    # Load best weights
    print(f"Loading best model from {_save_path}")
    model.load_state_dict(torch.load(str(_save_path), map_location=_device))

    return model, history


# ---------------------------------------------------------------------------
# Tokenizer Helper
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str = config.MODEL_NAME) -> PreTrainedTokenizerBase:
    """
    Load IndoBERT tokenizer.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.

    Returns
    -------
    PreTrainedTokenizerBase
    """
    return AutoTokenizer.from_pretrained(model_name)


def load_trained_model(
    weights_path: Optional[Path] = None,
    model_name: str = config.MODEL_NAME,
    device: Optional[torch.device] = None,
) -> IndoBERTClassifier:
    """
    Load a previously saved IndoBERTClassifier from weights file.

    Parameters
    ----------
    weights_path : Path, optional
        Path to .pt weights file. Defaults to config.BEST_MODEL_PATH.
    model_name : str
        Base model identifier (must match what was used for training).
    device : torch.device, optional

    Returns
    -------
    IndoBERTClassifier
        Model in eval mode.
    """
    _device = device or config.DEVICE
    _path = weights_path or config.BEST_MODEL_PATH

    model = IndoBERTClassifier(model_name, config.NUM_LABELS)
    model.load_state_dict(torch.load(str(_path), map_location=_device))
    model.to(_device)
    model.eval()
    return model


if __name__ == "__main__":
    import pandas as pd
    from src.preprocessing import load_dataset, preprocess_dataframe, remove_duplicates, split_data
    from src.dataset import build_dataloaders

    config.set_all_seeds()

    df = load_dataset()
    df = preprocess_dataframe(df)
    df = remove_duplicates(df)
    train_df, val_df, test_df = split_data(df)

    tokenizer = load_tokenizer()

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df["clean_text"].tolist(), train_df[config.LABEL_COL].tolist(),
        val_df["clean_text"].tolist(), val_df[config.LABEL_COL].tolist(),
        test_df["clean_text"].tolist(), test_df[config.LABEL_COL].tolist(),
        tokenizer,
    )

    model, history = fine_tune_indobert(train_loader, val_loader)
    print("\nTraining complete.")
    print(f"Best Val Macro F1: {max(history['val_macro_f1']):.4f}")
