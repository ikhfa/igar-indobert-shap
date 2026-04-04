"""
Data loading, cleaning, and preprocessing for Indonesian government app reviews.

Handles:
- CSV loading with column validation
- Indonesian slang normalization (100+ entries)
- Text cleaning (URLs, emojis, special chars)
- Deduplication
- Stratified splitting
- Synthetic sample generation for development
"""

import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ---------------------------------------------------------------------------
# Indonesian Slang Dictionary (100+ entries)
# ---------------------------------------------------------------------------

SLANG_DICT: Dict[str, str] = {
    # Pronouns / determiners
    "gue": "saya", "gw": "saya", "aku": "saya", "lo": "kamu", "lu": "kamu",
    "org": "orang", "mrk": "mereka",
    # Negation
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "enggak": "tidak",
    "ngga": "tidak", "ndak": "tidak", "g": "tidak",
    # Common verbs / connectors
    "udah": "sudah", "udh": "sudah", "dah": "sudah",
    "blm": "belum", "blom": "belum",
    "lg": "lagi", "lgi": "lagi",
    "bs": "bisa", "bsa": "bisa",
    "mau": "mau", "mo": "mau",
    "tau": "tahu", "tw": "tahu",
    "bilang": "bilang", "blng": "bilang",
    "dtg": "datang", "pergi": "pergi",
    "pke": "pakai", "pake": "pakai",
    "trs": "terus", "trus": "terus",
    "sampe": "sampai", "ampe": "sampai",
    "bgt": "banget", "bngt": "banget", "bngtt": "banget",
    "bener": "benar", "bnr": "benar",
    "emang": "memang", "emg": "memang",
    "kyk": "kayak", "kyak": "kayak", "kayak": "seperti",
    "gitu": "begitu", "gt": "begitu", "gini": "begini",
    "doang": "saja", "dong": "dong", "deh": "deh", "nih": "ini",
    "tuh": "itu", "tth": "itu",
    # Prepositions / conjunctions
    "yg": "yang", "yng": "yang",
    "krn": "karena", "karna": "karena", "krna": "karena",
    "dgn": "dengan", "dg": "dengan", "sm": "sama",
    "utk": "untuk", "buat": "untuk", "bwt": "untuk",
    "jg": "juga", "juga": "juga",
    "klo": "kalau", "kalu": "kalau", "kl": "kalau", "klw": "kalau",
    "tp": "tapi", "tapi": "tapi", "ttpi": "tapi",
    "spy": "supaya", "biar": "supaya",
    "dr": "dari", "dri": "dari",
    "ke": "ke", "pd": "pada",
    "sdh": "sudah", "sdah": "sudah",
    "skrg": "sekarang", "skrang": "sekarang",
    "tlng": "tolong", "mhon": "mohon",
    # Time
    "kmrn": "kemarin", "bsk": "besok", "hr": "hari", "bln": "bulan",
    "thn": "tahun", "mgg": "minggu",
    # App-specific
    "app": "aplikasi", "aplksi": "aplikasi", "apk": "aplikasi",
    "versi": "versi", "updt": "update", "upd": "update",
    "instal": "install", "unstall": "uninstall",
    "load": "muat", "loading": "memuat",
    "error": "error", "eror": "error",
    "login": "masuk", "log in": "masuk",
    "pass": "kata sandi", "password": "kata sandi",
    # Sentiment / adjectives
    "ok": "oke", "oke": "oke", "okey": "oke",
    "bagus": "bagus", "bgus": "bagus",
    "jelek": "jelek", "jlek": "jelek",
    "lambat": "lambat", "lemot": "lambat", "lmbt": "lambat",
    "cepet": "cepat", "cpat": "cepat",
    "susah": "sulit", "susa": "sulit",
    "mudah": "mudah", "muda": "mudah",
    "mahal": "mahal", "mhal": "mahal",
    "murah": "murah", "mrh": "murah",
    "baguss": "bagus", "bguss": "bagus",
    "parah": "parah", "prh": "parah",
    "keren": "keren", "krn2": "keren",
    "mantap": "mantap", "mantabs": "mantap", "mantep": "mantap",
    "aman": "aman", "amn": "aman",
    # Common typos / abbreviations
    "sdgkan": "sedangkan", "jdi": "jadi", "jd": "jadi",
    "hrs": "harus", "hrus": "harus",
    "byk": "banyak", "bnyk": "banyak",
    "sdikit": "sedikit", "sdikt": "sedikit",
    "dtny": "datanya", "dtnya": "datanya",
    "pls": "tolong", "please": "tolong",
    "thx": "terima kasih", "makasih": "terima kasih", "mksh": "terima kasih",
    "thanks": "terima kasih", "tq": "terima kasih",
}


# ---------------------------------------------------------------------------
# Synthetic Sample Generator
# ---------------------------------------------------------------------------

_SAMPLE_POSITIVES = [
    "aplikasi ini sangat bagus dan mudah digunakan",
    "pelayanan sangat memuaskan terima kasih",
    "sangat membantu pekerjaan sehari hari",
    "fitur lengkap dan tidak ada masalah",
    "update terbaru semakin baik dan lancar",
    "antarmuka ramah pengguna dan responsif",
    "proses cepat dan akurat tidak ada error",
    "sangat membantu untuk urusan administrasi",
    "aplikasi berjalan dengan lancar dan stabil",
    "mudah dipahami bahkan untuk orang awam",
]
_SAMPLE_NEUTRALS = [
    "biasa saja tidak ada yang istimewa",
    "cukup membantu tapi masih ada kekurangan",
    "lumayan bagus tapi perlu peningkatan",
    "fitur standar tidak lebih tidak kurang",
    "bisa digunakan tapi agak lambat",
    "ok lah untuk kebutuhan dasar",
    "cukup fungsional tapi tampilannya kurang menarik",
    "belum mencoba semua fitur tapi sepertinya oke",
    "tidak ada masalah besar tapi juga tidak sempurna",
    "lumayan untuk aplikasi pemerintah",
]
_SAMPLE_NEGATIVES = [
    "aplikasi sering error dan tidak bisa login",
    "sangat lambat bahkan untuk membuka halaman utama",
    "crash terus menerus sangat mengecewakan",
    "tidak bisa upload dokumen sudah berulang kali coba",
    "server selalu down saat dibutuhkan",
    "data tidak tersimpan dengan benar sangat frustrasi",
    "update malah membuat aplikasi semakin buruk",
    "tidak ada respons dari customer service",
    "bug masih banyak dan tidak diperbaiki",
    "gagal terus padahal sudah update ke versi terbaru",
]
_APP_NAMES = [
    "MyGov", "BPJS Kesehatan", "Samsat Online", "e-KTP", "SIAP",
    "Antrian Online", "PeduliLindungi", "eHAC", "BRImo", "Simpel",
]


def generate_sample_dataset(
    n: int = config.SAMPLE_SIZE,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic Indonesian-like review dataset for development.

    Produces balanced classes with minor noise (repeated chars, mixed slang).

    Parameters
    ----------
    n : int
        Total number of records to generate.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: review_text, label, app_name, rating.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    per_class = n // 3
    extras = n - per_class * 3

    records = []
    pools = [_SAMPLE_NEGATIVES, _SAMPLE_NEUTRALS, _SAMPLE_POSITIVES]
    rating_ranges = [(1, 2), (3, 3), (4, 5)]

    for label, (pool, (rlo, rhi)) in enumerate(zip(pools, rating_ranges)):
        count = per_class + (1 if label < extras else 0)
        for _ in range(count):
            base = rng.choice(pool)
            # Add minor noise
            if rng.random() < 0.3:
                words = base.split()
                idx = rng.randint(0, len(words) - 1)
                words[idx] = words[idx] + rng.choice(["", "nya", " banget", " sekali"])
                base = " ".join(words)
            records.append({
                config.TEXT_COL: base,
                config.LABEL_COL: label,
                config.APP_COL: rng.choice(_APP_NAMES),
                config.RATING_COL: int(np_rng.integers(rlo, rhi + 1)),
            })

    df = pd.DataFrame(records)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(
    path: Optional[str] = None,
    use_sample: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Load the review dataset from CSV or generate a synthetic sample.

    Parameters
    ----------
    path : str, optional
        Path to CSV file. Defaults to config.DATASET_PATH.
    use_sample : bool, optional
        Override config.USE_SAMPLE. If True, returns synthetic data.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with required columns.

    Raises
    ------
    FileNotFoundError
        If use_sample=False and file does not exist.
    ValueError
        If required columns are missing.
    """
    _use_sample = config.USE_SAMPLE if use_sample is None else use_sample
    _path = Path(path) if path else config.DATASET_PATH

    if _use_sample or not _path.exists():
        if not _use_sample:
            print(f"[Warning] Dataset not found at {_path}. Generating synthetic sample.")
        else:
            print("[Info] USE_SAMPLE=True — generating synthetic dataset.")
        return generate_sample_dataset()

    print(f"Loading dataset from {_path} ...")
    df = pd.read_csv(_path, low_memory=False)

    required = [config.TEXT_COL, config.LABEL_COL, config.APP_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df[config.LABEL_COL] = df[config.LABEL_COL].astype(int)
    df[config.TEXT_COL] = df[config.TEXT_COL].astype(str)
    print(f"Loaded {len(df):,} records.")
    return df


# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------

# Compiled patterns for speed
_URL_PAT = re.compile(r"https?://\S+|www\.\S+")
_EMOJI_PAT = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
_SPECIAL_PAT = re.compile(r"[^a-z0-9\s]")
_REPEAT_PAT = re.compile(r"(.)\1{2,}")
_SPACE_PAT = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Lowercase, strip URLs/emojis/special chars, normalize whitespace.

    Also normalizes repeated characters (e.g. 'bagussss' → 'baguss').

    Parameters
    ----------
    text : str
        Raw review text.

    Returns
    -------
    str
        Cleaned text.
    """
    text = str(text).lower()
    text = _URL_PAT.sub(" ", text)
    text = _EMOJI_PAT.sub(" ", text)
    text = _SPECIAL_PAT.sub(" ", text)
    text = _REPEAT_PAT.sub(r"\1\1", text)
    text = _SPACE_PAT.sub(" ", text).strip()
    return text


def normalize_slang(
    text: str,
    slang_dict: Optional[Dict[str, str]] = None,
) -> str:
    """
    Replace Indonesian slang terms with their formal equivalents.

    Tokenizes on whitespace, checks each token against the dictionary,
    and reconstructs the sentence. Case-insensitive matching.

    Parameters
    ----------
    text : str
        Pre-cleaned (lowercase) text.
    slang_dict : dict, optional
        Slang → formal mapping. Defaults to the built-in SLANG_DICT.

    Returns
    -------
    str
        Text with slang terms replaced.
    """
    _dict = slang_dict if slang_dict is not None else SLANG_DICT
    tokens = text.split()
    return " ".join(_dict.get(tok, tok) for tok in tokens)


def preprocess_text(text: str, slang_dict: Optional[Dict[str, str]] = None) -> str:
    """
    Full preprocessing pipeline: clean → normalize slang.

    Parameters
    ----------
    text : str
        Raw text.
    slang_dict : dict, optional
        Custom slang dictionary.

    Returns
    -------
    str
        Fully preprocessed text.
    """
    return normalize_slang(clean_text(text), slang_dict)


def preprocess_dataframe(
    df: pd.DataFrame,
    slang_dict: Optional[Dict[str, str]] = None,
    text_col: str = config.TEXT_COL,
    min_len: int = config.MIN_TEXT_LEN,
) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    slang_dict : dict, optional
        Slang dictionary override.
    text_col : str
        Column containing review text.
    min_len : int
        Minimum token count; rows with fewer tokens after cleaning are dropped.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame (new column 'clean_text' added).
    """
    df = df.copy()
    tqdm.pandas(desc="Preprocessing text")
    df["clean_text"] = df[text_col].progress_apply(
        lambda t: preprocess_text(t, slang_dict)
    )
    # Drop too-short reviews
    before = len(df)
    df = df[df["clean_text"].str.split().str.len() >= min_len].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with text shorter than {min_len} tokens.")
    return df


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def remove_duplicates(
    df: pd.DataFrame,
    text_col: str = "clean_text",
) -> pd.DataFrame:
    """
    Drop exact duplicate review texts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with preprocessed text.
    text_col : str
        Column to deduplicate on.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    print(f"Removed {before - len(df):,} duplicates. Remaining: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Stratified Sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    df: pd.DataFrame,
    n_per_app: int = config.SAMPLE_N_PER_APP,
    label_col: str = config.LABEL_COL,
    app_col: str = config.APP_COL,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Sample up to n_per_app records per (app, label) combination.

    Useful for reducing class imbalance when working with the full 617K dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    n_per_app : int
        Max records per app per sentiment class.
    label_col : str
        Sentiment label column.
    app_col : str
        Application name column.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Sampled DataFrame.
    """
    groups = df.groupby([app_col, label_col])
    sampled = groups.apply(
        lambda g: g.sample(min(len(g), n_per_app), random_state=seed)
    )
    result = sampled.reset_index(drop=True)
    print(f"Stratified sample: {len(result):,} records from {len(df):,}")
    return result


# ---------------------------------------------------------------------------
# Train / Val / Test Split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    train: float = config.TRAIN_SPLIT,
    val: float = config.VAL_SPLIT,
    test: float = config.TEST_SPLIT,
    seed: int = config.RANDOM_SEED,
    label_col: str = config.LABEL_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/val/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (after preprocessing).
    train : float
        Train fraction.
    val : float
        Validation fraction.
    test : float
        Test fraction.
    seed : int
        Random seed.
    label_col : str
        Column to stratify on.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"

    y = df[label_col].values

    # First split off test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(df, y))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Then split train/val from the remainder
    val_fraction = val / (train + val)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    y_tv = train_val_df[label_col].values
    train_idx, val_idx = next(sss2.split(train_val_df, y_tv))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    print(f"Split sizes — Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    config.set_all_seeds()

    df = load_dataset()
    print(df.head(3))
    print(f"\nLabel distribution:\n{df[config.LABEL_COL].value_counts()}")

    df = preprocess_dataframe(df)
    df = remove_duplicates(df)
    train_df, val_df, test_df = split_data(df)
    print(f"\nTrain label dist:\n{train_df[config.LABEL_COL].value_counts()}")
