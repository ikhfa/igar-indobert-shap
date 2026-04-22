"""
SHAP-based token attribution for IndoBERT sentiment classification.

Uses shap.Explainer with the HuggingFace text-classification pipeline wrapper.
Subword SHAP values are aggregated to word level for interpretability.
"""

import hashlib
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

tqdm = config.get_tqdm()


# ---------------------------------------------------------------------------
# Pipeline Wrapper for SHAP
# ---------------------------------------------------------------------------

class _HFPipelineWrapper:
    """
    Wraps IndoBERTClassifier as a callable that returns probability arrays,
    compatible with shap.Explainer's text pipeline interface.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        batch_size: int = config.SHAP_BATCH_SIZE,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.model.eval()

    def __call__(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities for a list of texts.

        Parameters
        ----------
        texts : List[str]

        Returns
        -------
        np.ndarray
            Shape (N, num_labels) probability matrix.
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(t) if not isinstance(t, str) else t for t in texts]
        all_probs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            enc = self.tokenizer(
                batch,
                max_length=config.MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model.bert(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)


# ---------------------------------------------------------------------------
# SHAP Explainer
# ---------------------------------------------------------------------------

class SHAPExplainer:
    """
    SHAP-based token importance explainer for IndoBERT.

    Uses shap.Explainer (partition/text masker) for model-agnostic attributions.
    Subword tokens are aggregated to word level by summing attribution values.

    Parameters
    ----------
    model : IndoBERTClassifier
        Fine-tuned IndoBERT model.
    tokenizer : PreTrainedTokenizerBase
        IndoBERT tokenizer.
    device : torch.device
    batch_size : int
        Batch size for model inference during SHAP.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[torch.device] = None,
        batch_size: int = config.SHAP_BATCH_SIZE,
    ) -> None:
        import shap

        self.model = model
        self.tokenizer = tokenizer
        self.device = device or config.DEVICE
        self.batch_size = batch_size

        self._wrapper = _HFPipelineWrapper(model, tokenizer, self.device, batch_size)

        masker = shap.maskers.Text(self.tokenizer)
        self.explainer = shap.Explainer(
            self._wrapper,
            masker=masker,
            output_names=config.LABEL_NAMES,
        )

    def _preprocess_for_shap(self, text: str) -> str:
        from src.preprocessing import preprocess_text
        return preprocess_text(text)

    @staticmethod
    def _cache_key(text: str, max_evals: int) -> str:
        return hashlib.md5(f"{text}:{max_evals}".encode()).hexdigest()

    def _get_cached(self, text: str, max_evals: int):
        cache_path = config.SHAP_CACHE_DIR / f"{self._cache_key(text, max_evals)}.pkl"
        if cache_path.exists():
            return pickle.loads(cache_path.read_bytes())
        return None

    def _set_cached(self, text: str, max_evals: int, result):
        config.SHAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = config.SHAP_CACHE_DIR / f"{self._cache_key(text, max_evals)}.pkl"
        cache_path.write_bytes(pickle.dumps(result))

    @staticmethod
    def _aggregate_subwords_to_words(
        tokens: list,
        shap_values: np.ndarray,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Aggregate subword token SHAP values to word-level SHAP values.

        Subword tokens prefixed with '##' (e.g. '##ja', '##kan') are summed
        into their parent word. Tokens without '##' start a new word.

        Parameters
        ----------
        tokens : list of str
            Subword tokens from the model tokenizer.
        shap_values : np.ndarray
            Shape (num_tokens, num_classes) SHAP values.

        Returns
        -------
        Tuple[List[str], np.ndarray]
            (word_list, word_shap_values) where word_shap_values has shape
            (num_words, num_classes).
        """
        if len(tokens) == 0:
            return [], np.array([]).reshape(0, shap_values.shape[1] if shap_values.ndim > 1 else 0)

        words: List[str] = []
        word_shap: List[np.ndarray] = []

        current_word = ""
        current_shap = np.zeros(shap_values.shape[1] if shap_values.ndim > 1 else 1, dtype=np.float64)

        for i, tok in enumerate(tokens):
            val = shap_values[i] if shap_values.ndim > 1 else np.array([shap_values[i]])
            if tok.startswith("##") and current_word:
                current_word += tok[2:]
                current_shap += val
            else:
                if current_word:
                    words.append(current_word)
                    word_shap.append(current_shap)
                current_word = tok.replace("##", "")
                current_shap = np.array(val, dtype=np.float64)

        if current_word:
            words.append(current_word)
            word_shap.append(current_shap)

        return words, np.array(word_shap)

    def explain_batch(
        self,
        texts: List[str],
        max_evals: int = config.SHAP_MAX_EVALS,
    ) -> List[Dict[str, Dict[str, float]]]:
        """
        Compute SHAP values for a batch of texts.

        Texts are preprocessed before SHAP analysis to ensure the masker
        operates on the same text the model sees. Subword tokens are
        aggregated to word level for human-readable attributions.

        Parameters
        ----------
        texts : List[str]
            Input texts to explain.
        max_evals : int
            Maximum model evaluations per sample for SHAP partition.

        Returns
        -------
        List[Dict[str, Dict[str, float]]]
            Each item: {class_name: {word: shap_value, ...}, ...}
        """
        preprocessed = [self._preprocess_for_shap(t) for t in texts]
        shap_values = self.explainer(preprocessed, max_evals=max_evals, silent=True)
        results = []

        for sample_idx in range(len(texts)):
            raw_tokens = list(shap_values.data[sample_idx])
            raw_values = shap_values.values[sample_idx]

            words, word_vals = self._aggregate_subwords_to_words(raw_tokens, raw_values)

            sample_result: Dict[str, Dict[str, float]] = {}
            for class_idx, class_name in enumerate(config.LABEL_NAMES):
                class_vals = word_vals[:, class_idx] if word_vals.ndim > 1 else word_vals
                sample_result[class_name] = {
                    str(word): float(val)
                    for word, val in zip(words, class_vals)
                    if str(word).strip()
                }
            results.append(sample_result)

        return results

    def explain_single(
        self,
        text: str,
        max_evals: int = config.SHAP_MAX_EVALS,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute SHAP values for a single review text.

        Uses disk caching to avoid recomputation.

        Parameters
        ----------
        text : str
            Input text.
        max_evals : int

        Returns
        -------
        dict
            {class_name: {word: shap_value}}
        """
        cached = self._get_cached(text, max_evals)
        if cached is not None:
            return cached
        results = self.explain_batch([text], max_evals=max_evals)
        self._set_cached(text, max_evals, results[0])
        return results[0]

    def aggregate_token_importance(
        self,
        shap_values_list: List[Dict[str, Dict[str, float]]],
        class_name: Optional[str] = None,
        top_n: int = config.SHAP_TOP_N_TOKENS,
    ) -> pd.DataFrame:
        """
        Aggregate SHAP values across multiple samples into a top-N token ranking.

        For each class, tokens are ranked by mean |SHAP value| across all samples.

        Parameters
        ----------
        shap_values_list : List[Dict]
            Output of explain_batch().
        class_name : str, optional
            Class to aggregate for. If None, aggregates all classes.
        top_n : int
            Number of top tokens to return per class.

        Returns
        -------
        pd.DataFrame
            Columns: token, mean_shap, abs_mean_shap, class_name, rank.
        """
        class_names = [class_name] if class_name else config.LABEL_NAMES
        rows = []

        for cls in class_names:
            token_sums: Dict[str, List[float]] = {}
            for sample in shap_values_list:
                for word, val in sample.get(cls, {}).items():
                    token_sums.setdefault(word, []).append(val)

            sorted_tokens = sorted(
                token_sums.items(),
                key=lambda kv: abs(np.mean(kv[1])),
                reverse=True,
            )[:top_n]

            for rank, (token, vals) in enumerate(sorted_tokens, 1):
                rows.append({
                    "class_name": cls,
                    "rank": rank,
                    "token": token,
                    "mean_shap": float(np.mean(vals)),
                    "abs_mean_shap": float(abs(np.mean(vals))),
                    "n_occurrences": len(vals),
                })

        return pd.DataFrame(rows)

    def plot_shap_summary(
        self,
        token_df: pd.DataFrame,
        class_name: str,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Horizontal bar chart of top-N SHAP tokens for a given class.

        Positive SHAP = supports the class prediction (green).
        Negative SHAP = opposes it (red).

        Parameters
        ----------
        token_df : pd.DataFrame
            Output of aggregate_token_importance() (filtered to one class).
        class_name : str
            Class name for title/labels.
        save_path : str or Path, optional
        title : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = token_df[token_df["class_name"] == class_name].head(config.SHAP_TOP_N_TOKENS)
        if df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No SHAP data", ha="center")
            return fig

        tokens = df["token"].tolist()
        values = df["mean_shap"].tolist()
        colors = ["#27ae60" if v > 0 else "#e74c3c" for v in values]

        fig, ax = plt.subplots(figsize=(9, max(5, len(tokens) * 0.42)))
        y_pos = range(len(tokens))
        ax.barh(list(y_pos), values, color=colors)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(tokens, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean SHAP Value")
        ax.set_title(title or f"Top SHAP Tokens — Class: {class_name}", fontsize=12)
        ax.invert_yaxis()
        plt.tight_layout()

        if save_path:
            fig.savefig(str(save_path), dpi=config.PLOT_DPI, bbox_inches="tight")
        return fig

    def plot_waterfall(
        self,
        text: str,
        class_name: str,
        save_path: Optional[Union[str, Path]] = None,
        max_evals: int = config.SHAP_MAX_EVALS,
    ) -> plt.Figure:
        """
        Waterfall-style contribution chart for a single review.

        Shows cumulative SHAP contributions from baseline to final prediction.

        Parameters
        ----------
        text : str
            Single review to explain.
        class_name : str
            Target class.
        save_path : str or Path, optional
        max_evals : int

        Returns
        -------
        matplotlib.figure.Figure
        """
        shap_data = self.explain_single(text, max_evals=max_evals)
        class_shap = shap_data.get(class_name, {})

        if not class_shap:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No SHAP data", ha="center")
            return fig

        # Sort by absolute value descending
        items = sorted(class_shap.items(), key=lambda x: abs(x[1]), reverse=True)
        tokens = [i[0] for i in items[:config.WATERFALL_TOP_N]]
        values = [i[1] for i in items[:config.WATERFALL_TOP_N]]

        # Compute cumulative
        probs_base = self._wrapper([self._preprocess_for_shap(text)])[0]
        class_idx = config.LABEL_NAMES.index(class_name)
        base_val = float(probs_base[class_idx])

        cumulative = [base_val]
        for v in values:
            cumulative.append(cumulative[-1] + v)

        fig, ax = plt.subplots(figsize=(10, max(5, len(tokens) * 0.5)))
        colors = ["#27ae60" if v > 0 else "#e74c3c" for v in values]

        y_pos = range(len(tokens))
        ax.barh(list(y_pos), values, color=colors)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(tokens, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value")
        ax.set_title(
            f"SHAP Waterfall — Class: {class_name}\n\"{text[:70]}{'...' if len(text) > 70 else ''}\"",
            fontsize=11,
        )
        ax.invert_yaxis()
        plt.tight_layout()

        if save_path:
            fig.savefig(str(save_path), dpi=config.PLOT_DPI, bbox_inches="tight")
        return fig


if __name__ == "__main__":
    from src.train import IndoBERTClassifier, load_tokenizer, load_trained_model

    config.set_all_seeds()

    # Load model (requires trained weights)
    tokenizer = load_tokenizer()
    try:
        model = load_trained_model()
        print("Loaded trained model.")
    except FileNotFoundError:
        print("No checkpoint found — using fresh model for demo.")
        model = IndoBERTClassifier()
        model.to(config.DEVICE)
        model.eval()

    explainer = SHAPExplainer(model, tokenizer, config.DEVICE)

    texts = [
        "aplikasi ini bagus sekali sangat membantu pekerjaan",
        "server selalu down saat dibutuhkan sangat mengecewakan",
    ]
    print("Running SHAP explain_batch...")
    shap_results = explainer.explain_batch(texts)

    for i, result in enumerate(shap_results):
        print(f"\nText {i+1}:")
        for cls_name, token_vals in result.items():
            top = sorted(token_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            print(f"  {cls_name}: {top}")

    df_agg = explainer.aggregate_token_importance(shap_results)
    print("\nTop aggregate tokens:")
    print(df_agg.head(10).to_string(index=False))
