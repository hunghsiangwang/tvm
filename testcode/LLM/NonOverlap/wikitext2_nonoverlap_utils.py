"""Shared utilities for the fixed WikiText-2 non-overlapping benchmark.

The sibling sliding-window evaluator in ``../Sliding_Window`` is intentionally separate:
it uses overlap as context and is suited to reference PPL sanity checks.  This
module implements the cheaper fixed-subset protocol used for datatype
comparisons; PPL values from the two protocols are not directly comparable.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


WINDOW_LENGTH = 512
SUBSET_SIZE = 8
SEED = 0
NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64
VOCAB_SIZE = 50257


def tokenize_text(text_path, tokenizer):
    """Tokenize the complete raw file as one deterministic GPT-2 stream."""

    text = Path(text_path).read_text(encoding="utf-8")
    old_limit = tokenizer.model_max_length
    tokenizer.model_max_length = max(old_limit, len(text) + 1)
    try:
        ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
    finally:
        tokenizer.model_max_length = old_limit
    return text, np.asarray(ids, dtype=np.int64)


def metadata_path(windows_path):
    return Path(windows_path).with_suffix(".meta")


def create_fixed_windows(text_path, tokenizer, windows_path):
    """Create the canonical L=512, seed=0, eight-window subset."""

    text, token_ids = tokenize_text(text_path, tokenizer)
    available = token_ids.size // WINDOW_LENGTH
    if available < SUBSET_SIZE:
        raise ValueError(
            f"WikiText-2 provides only {available} complete windows; need {SUBSET_SIZE}"
        )
    all_windows = token_ids[: available * WINDOW_LENGTH].reshape(
        available, WINDOW_LENGTH
    )
    rng = np.random.default_rng(SEED)
    selected_indices = np.sort(
        rng.choice(available, size=SUBSET_SIZE, replace=False)
    )
    selected = np.ascontiguousarray(all_windows[selected_indices], dtype=np.int64)

    windows_path = Path(windows_path)
    if windows_path.suffix != ".npy":
        raise ValueError("--windows-file must end in .npy")
    windows_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(windows_path, selected, allow_pickle=False)
    metadata = {
        "source": Path(text_path).name,
        "source_sha256": hashlib.sha256(Path(text_path).read_bytes()).hexdigest(),
        "tokenizer": Path(str(getattr(tokenizer, "name_or_path", "GPT-2"))).name,
        "total_raw_characters": len(text),
        "total_gpt2_tokens": int(token_ids.size),
        "window_length": WINDOW_LENGTH,
        "total_available_windows": int(available),
        "seed": SEED,
        "selected_indices": selected_indices.tolist(),
        "number_of_windows": SUBSET_SIZE,
        "input_tokens": SUBSET_SIZE * WINDOW_LENGTH,
        "scored_tokens": SUBSET_SIZE * (WINDOW_LENGTH - 1),
        "windows_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
    }
    metadata_path(windows_path).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return selected, metadata


def load_or_create_fixed_windows(
    text_path, tokenizer, windows_path, regenerate=False
):
    """Load the immutable subset by default, creating it only when absent."""

    windows_path = Path(windows_path)
    sidecar = metadata_path(windows_path)
    if regenerate or not windows_path.exists():
        return create_fixed_windows(text_path, tokenizer, windows_path)

    windows = np.load(windows_path, allow_pickle=False)
    if windows.shape != (SUBSET_SIZE, WINDOW_LENGTH) or windows.dtype != np.int64:
        raise ValueError(
            f"{windows_path} must have shape {(SUBSET_SIZE, WINDOW_LENGTH)} "
            f"and dtype int64, got {windows.shape} and {windows.dtype}"
        )
    if not sidecar.exists():
        raise FileNotFoundError(
            f"Missing benchmark metadata {sidecar}; use --regenerate-windows once"
        )
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    required = {
        "window_length": WINDOW_LENGTH,
        "seed": SEED,
        "number_of_windows": SUBSET_SIZE,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Invalid {sidecar}: {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    digest = hashlib.sha256(np.ascontiguousarray(windows).tobytes()).hexdigest()
    if digest != metadata.get("windows_sha256"):
        raise ValueError(f"{windows_path} does not match the SHA-256 in {sidecar}")
    return windows, metadata


def print_dataset_summary(metadata, windows_path, num_windows):
    selected = metadata["selected_indices"]
    print("Dataset:")
    print(f"  Source: {metadata['source']}")
    print(f"  Tokenizer: {metadata['tokenizer']}")
    print(f"  Total raw characters: {metadata['total_raw_characters']}")
    print(f"  Total GPT-2 tokens: {metadata['total_gpt2_tokens']}")
    print(f"  Total available 512-token windows: {metadata['total_available_windows']}")
    print(f"  Window Length: {metadata['window_length']}")
    print(f"  Seed: {metadata['seed']}")
    print(f"  Selected window indices: {selected}")
    print(f"  Fixed windows file: {Path(windows_path).resolve()}")
    print(f"  Number of Windows: {num_windows}")
    print(f"  Input Tokens: {num_windows * WINDOW_LENGTH}")
    print(f"  Scored Tokens: {num_windows * (WINDOW_LENGTH - 1)}")


def stable_nll_sum(logits, labels, chunk_rows=16):
    """Sum NLL with chunked float64 logsumexp and no full-size temporary."""

    total = 0.0
    for start in range(0, labels.size, chunk_rows):
        stop = min(start + chunk_rows, labels.size)
        values = np.asarray(logits[start:stop], dtype=np.float64)
        row_max = np.max(values, axis=1)
        lse = row_max + np.log(
            np.sum(np.exp(values - row_max[:, None]), axis=1, dtype=np.float64)
        )
        targets = values[np.arange(stop - start), labels[start:stop]]
        total += float(np.sum(lse - targets, dtype=np.float64))
    return total


def compare_logits(lhs, rhs, chunk_rows=16):
    """Return absolute-error sum/count and scored-position argmax matches."""

    if lhs.shape != rhs.shape:
        raise ValueError(f"Cannot compare logits shapes {lhs.shape} and {rhs.shape}")
    absolute_sum = 0.0
    value_count = 0
    matches = 0
    for start in range(0, lhs.shape[0], chunk_rows):
        stop = min(start + chunk_rows, lhs.shape[0])
        left = np.asarray(lhs[start:stop], dtype=np.float32)
        right = np.asarray(rhs[start:stop], dtype=np.float32)
        absolute_sum += float(
            np.sum(np.abs(left - right), dtype=np.float64)
        )
        value_count += left.size
        matches += int(
            np.count_nonzero(np.argmax(left, axis=1) == np.argmax(right, axis=1))
        )
    return absolute_sum, value_count, matches
