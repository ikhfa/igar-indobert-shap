"""
PyTorch Dataset class for IndoBERT fine-tuning.

Handles tokenization, padding, and truncation for sequence classification.
Uses lazy (on-demand) tokenization to reduce peak memory usage.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class IndoBERTDataset(Dataset):
    """
    PyTorch Dataset wrapping Indonesian reviews for IndoBERT.

    Uses lazy tokenization: texts are tokenized on-demand in __getitem__
    instead of all at once in __init__, reducing peak memory from ~50GB to
    negligible.

    Parameters
    ----------
    texts : List[str]
        Preprocessed review texts.
    labels : List[int]
        Integer sentiment labels (0=Negative, 1=Neutral, 2=Positive).
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer for IndoBERT.
    max_len : int
        Maximum sequence length (tokens). Sequences are truncated/padded to this length.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = config.MAX_LEN,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(
                f"texts ({len(texts)}) and labels ({len(labels)}) must have equal length."
            )
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return a single encoded sample as a dict of tensors.

        Tokenizes the text on-demand (lazy tokenization).

        Returns
        -------
        dict
            Keys: input_ids, attention_mask, token_type_ids, labels.
        """
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids",
                torch.zeros(self.max_len, dtype=torch.long)
            ).squeeze(0) if "token_type_ids" in encoding else torch.zeros(self.max_len, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    @property
    def num_classes(self) -> int:
        """Number of unique classes in this dataset."""
        return len(set(self.labels))


def build_dataloaders(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = config.BATCH_SIZE,
    eval_batch_size: int = config.EVAL_BATCH_SIZE,
    max_len: int = config.MAX_LEN,
) -> tuple:
    """
    Build train, val, and test DataLoaders from text lists.

    Parameters
    ----------
    train_texts, val_texts, test_texts : List[str]
        Preprocessed texts for each split.
    train_labels, val_labels, test_labels : List[int]
        Corresponding labels.
    tokenizer : PreTrainedTokenizerBase
        IndoBERT tokenizer.
    batch_size : int
        Training batch size.
    eval_batch_size : int
        Validation/test batch size.
    max_len : int
        Sequence length.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        (train_loader, val_loader, test_loader)
    """
    train_ds = IndoBERTDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds = IndoBERTDataset(val_texts, val_labels, tokenizer, max_len)
    test_ds = IndoBERTDataset(test_texts, test_labels, tokenizer, max_len)

    g = torch.Generator()
    g.manual_seed(config.RANDOM_SEED)

    num_workers = config.DATALOADER_NUM_WORKERS
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from transformers import AutoTokenizer

    config.set_all_seeds()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    texts = ["aplikasi ini bagus sekali", "sangat lambat dan error terus"]
    labels = [2, 0]

    ds = IndoBERTDataset(texts, labels, tokenizer)
    sample = ds[0]
    print("Keys:", list(sample.keys()))
    print("input_ids shape:", sample["input_ids"].shape)
    print("label:", sample["labels"].item(), "→", config.LABEL_MAP[sample["labels"].item()])
