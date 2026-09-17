import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tvm


SCRIPT_DIR = Path(__file__).resolve().parent
LLM_DIR = SCRIPT_DIR.parent
DEFAULT_TOKENIZER = str(LLM_DIR / "gpt2-ONNX")
DEFAULT_HF_MODEL = "openai-community/gpt2"
NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64
VOCAB_SIZE = 50257


@dataclass(frozen=True)
class EvaluationWindow:
    start: int
    end: int
    score_start: int

    @property
    def scored_tokens(self):
        return self.end - self.score_start


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def nonnegative_float(value):
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def percentage(value):
    value = float(value)
    if not 0 <= value <= 100:
        raise argparse.ArgumentTypeError("value must be between 0 and 100")
    return value


def resolve_dataset_path(requested_path):
    requested = Path(requested_path).expanduser()
    roots = [requested]
    if not requested.is_absolute():
        roots.append(SCRIPT_DIR / requested)
        roots.append(LLM_DIR / requested)

    # Keep the downloaded helper's existing output as a compatibility fallback.
    roots.append(LLM_DIR / "wikitext2_test.txt")
    for root in roots:
        if root.is_file():
            return root.resolve()
        if not root.is_dir():
            continue
        preferred = (
            root / "wiki.test.raw",
            root / "test",
            root / "test.txt",
            root / "wikitext2_test.txt",
        )
        for candidate in preferred:
            if candidate.is_file():
                return candidate.resolve()
        matches = sorted(
            candidate
            for pattern in ("**/wiki.test.raw", "**/*test*.raw", "**/*test*.txt")
            for candidate in root.glob(pattern)
            if candidate.is_file()
        )
        if matches:
            return matches[0].resolve()
    raise FileNotFoundError(
        f"Cannot find WikiText-2 test text from {requested_path!r}. "
        "Pass --dataset-path with the raw test file or its containing directory."
    )


def tokenize_corpus(dataset_path, tokenizer):
    # Read the raw file as one string so every newline and article boundary stays
    # in its original deterministic order.
    text = dataset_path.read_text(encoding="utf-8")
    # This call intentionally tokenizes a corpus much longer than GPT-2's model
    # context.  Disable only the tokenizer's advisory length warning; individual
    # model calls still obey sequence_length and the model position limit.
    original_model_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = max(original_model_max_length, len(text) + 1)
    try:
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
    finally:
        tokenizer.model_max_length = original_model_max_length
    return np.asarray(token_ids, dtype=np.int64)


def build_sliding_windows(token_count, sequence_length, stride):
    """Cover targets 1..token_count-1 once, using overlap only as context."""

    if token_count < sequence_length:
        raise ValueError(
            f"evaluation token count ({token_count}) must be at least the compiled "
            f"sequence length ({sequence_length})"
        )
    if stride > sequence_length:
        raise ValueError("stride must not exceed sequence length")

    windows = []
    start = 0
    previous_end = 0
    while True:
        end = start + sequence_length
        score_start = 1 if not windows else previous_end
        windows.append(EvaluationWindow(start, end, score_start))
        if end == token_count:
            break
        previous_end = end
        start = min(start + stride, token_count - sequence_length)

    total_scored = sum(window.scored_tokens for window in windows)
    if total_scored != token_count - 1:
        raise AssertionError(
            f"sliding-window mask scores {total_scored} tokens; expected {token_count - 1}"
        )
    return windows


def scored_logits_and_labels(logits, input_ids, window):
    """Apply causal shift, then retain only this window's newly covered targets."""

    label_start = window.score_start - window.start
    # local target j is predicted by local logits j-1.  label_start is always
    # at least one, so no window attempts to score a target without left context.
    scored_logits = logits[0, label_start - 1 : -1, :]
    scored_labels = input_ids[0, label_start:]
    if scored_logits.shape[0] != window.scored_tokens:
        raise AssertionError(
            f"window produced {scored_logits.shape[0]} scored rows; "
            f"expected {window.scored_tokens}"
        )
    return scored_logits, scored_labels


def stable_nll_sum(logits, labels):
    """Sum cross entropy in float64 without explicitly forming softmax."""

    values = np.asarray(logits, dtype=np.float64)
    row_max = np.max(values, axis=-1)
    logsumexp = row_max + np.log(
        np.sum(np.exp(values - row_max[:, None]), axis=-1, dtype=np.float64)
    )
    target_logits = values[np.arange(labels.shape[0]), labels]
    return float(np.sum(logsumexp - target_logits, dtype=np.float64))


def make_tvm_inputs(input_ids, device):
    sequence_length = input_ids.shape[1]
    tensors = [tvm.runtime.tensor(input_ids, device=device)]
    empty_cache = np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32)
    tensors.extend(
        tvm.runtime.tensor(empty_cache, device=device) for _ in range(NUM_LAYERS * 2)
    )
    tensors.append(
        tvm.runtime.tensor(np.ones((1, sequence_length), dtype=np.int64), device=device)
    )
    tensors.append(
        tvm.runtime.tensor(
            np.arange(sequence_length, dtype=np.int64)[None, :], device=device
        )
    )
    return tensors


def synchronize(device):
    # CPU calls are normally synchronous, but an explicit sync keeps timing valid
    # for TVM device implementations that enqueue work.
    if hasattr(device, "sync"):
        device.sync()


def validate_hf_config(config, sequence_length):
    actual = {
        "layers": config.n_layer,
        "heads": config.n_head,
        "head_dim": config.n_embd // config.n_head,
        "vocab_size": config.vocab_size,
    }
    expected = {
        "layers": NUM_LAYERS,
        "heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB_SIZE,
    }
    if actual != expected:
        raise ValueError(f"HF model config {actual} does not match ONNX GPT-2 Small {expected}")
    if sequence_length > config.n_positions:
        raise ValueError(
            f"sequence length {sequence_length} exceeds model limit {config.n_positions}"
        )


def load_reference_model(model_name):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as err:
        raise RuntimeError(
            "HF FP32 equivalence requires PyTorch and Transformers. "
            "Install the PyTorch build appropriate for this machine first."
        ) from err

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device="cpu", dtype=torch.float32)
    model.eval()
    return torch, model


def evaluate(args):
    try:
        from transformers import AutoTokenizer
    except ImportError as err:
        raise RuntimeError("Transformers is required for the GPT-2 tokenizer") from err

    if args.num_threads is not None:
        os.environ["TVM_NUM_THREADS"] = str(args.num_threads)

    dataset_path = resolve_dataset_path(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    all_token_ids = tokenize_corpus(dataset_path, tokenizer)
    if args.evaluation_tokens == 0:
        evaluation_ids = all_token_ids
    else:
        if args.evaluation_tokens < args.sequence_length:
            raise ValueError(
                "--evaluation-tokens must be at least --sequence-length"
            )
        if all_token_ids.size < args.evaluation_tokens:
            raise ValueError(
                f"WikiText-2 test has {all_token_ids.size} tokens, "
                f"fewer than the requested {args.evaluation_tokens}"
            )

        evaluation_ids = all_token_ids[:args.evaluation_tokens]
    windows = build_sliding_windows(
        len(evaluation_ids), args.sequence_length, args.stride
    )
    expected_scored_tokens = len(evaluation_ids) - 1

    print("WikiText-2 preprocessing:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Total tokenized WikiText-2 test tokens: {len(all_token_ids)}")
    print(f"  Evaluation token count: {len(evaluation_ids)}")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Stride: {args.stride}")
    print(f"  Number of windows: {len(windows)}")

    torch, hf_model = load_reference_model(args.hf_model)
    validate_hf_config(hf_model.config, args.sequence_length)

    device = tvm.cpu()
    lib = tvm.runtime.load_module(args.model)
    decoder_model = tvm.relax.vm.VirtualMachine(lib, device)

    first_ids = evaluation_ids[windows[0].start : windows[0].end][None, :]
    warmup_inputs = make_tvm_inputs(first_ids, device)
    print(f"TVM warm-up: {args.warmup_windows} window(s), excluded from timing")
    for _ in range(args.warmup_windows):
        decoder_model["main"](*warmup_inputs)
        synchronize(device)

    hf_total_nll = 0.0
    tvm_total_nll = 0.0
    total_scored_tokens = 0
    total_abs_logit_error = 0.0
    compared_logit_values = 0
    maximum_window_mae = 0.0
    maximum_absolute_logit_error = 0.0
    matched_tokens = 0
    tvm_forward_times = []

    evaluation_start = time.perf_counter()
    for index, window in enumerate(windows, start=1):
        window_ids = evaluation_ids[window.start : window.end][None, :]
        attention_mask = np.ones((1, args.sequence_length), dtype=np.int64)
        position_ids = np.arange(args.sequence_length, dtype=np.int64)[None, :]

        with torch.inference_mode():
            hf_output = hf_model(
                input_ids=torch.from_numpy(window_ids),
                attention_mask=torch.from_numpy(attention_mask),
                position_ids=torch.from_numpy(position_ids),
                use_cache=False,
            )
        hf_logits = hf_output.logits.detach().cpu().float().numpy()

        tvm_inputs = make_tvm_inputs(window_ids, device)
        forward_start = time.perf_counter()
        tvm_output = decoder_model["main"](*tvm_inputs)
        synchronize(device)
        tvm_forward_times.append(time.perf_counter() - forward_start)
        tvm_logits = tvm_output[0].numpy()

        expected_shape = (1, args.sequence_length, VOCAB_SIZE)
        if hf_logits.shape != expected_shape or tvm_logits.shape != expected_shape:
            raise ValueError(
                f"Unexpected logits shapes: HF={hf_logits.shape}, TVM={tvm_logits.shape}, "
                f"expected={expected_shape}. Check the compiled --sequence-length."
            )

        hf_scored, labels = scored_logits_and_labels(hf_logits, window_ids, window)
        tvm_scored, tvm_labels = scored_logits_and_labels(tvm_logits, window_ids, window)
        if not np.array_equal(labels, tvm_labels):
            raise AssertionError("HF and TVM scoring labels differ")

        hf_total_nll += stable_nll_sum(hf_scored, labels)
        tvm_total_nll += stable_nll_sum(tvm_scored, labels)
        total_scored_tokens += labels.size

        absolute_error = np.abs(hf_scored - tvm_scored)
        window_mae = float(np.mean(absolute_error, dtype=np.float64))
        total_abs_logit_error += float(np.sum(absolute_error, dtype=np.float64))
        compared_logit_values += absolute_error.size
        maximum_window_mae = max(maximum_window_mae, window_mae)
        maximum_absolute_logit_error = max(
            maximum_absolute_logit_error, float(np.max(absolute_error))
        )

        hf_predictions = np.argmax(hf_scored, axis=-1)
        tvm_predictions = np.argmax(tvm_scored, axis=-1)
        window_matches = int(np.count_nonzero(hf_predictions == tvm_predictions))
        matched_tokens += window_matches
        print(
            f"Window {index:2d}/{len(windows)}: tokens [{window.start}:{window.end}], "
            f"scored targets [{window.score_start}:{window.end}] "
            f"({labels.size}), logits MAE={window_mae:.6e}, "
            f"argmax match={100.0 * window_matches / labels.size:.2f}%"
        )

        # Release the full vocabulary tensors before the next window.
        del hf_output, hf_logits, tvm_output, tvm_logits, absolute_error

    end_to_end_evaluation_time = time.perf_counter() - evaluation_start
    if total_scored_tokens != expected_scored_tokens:
        raise AssertionError(
            f"scored {total_scored_tokens} targets; expected {expected_scored_tokens}"
        )

    hf_ppl = float(np.exp(hf_total_nll / total_scored_tokens))
    tvm_ppl = float(np.exp(tvm_total_nll / total_scored_tokens))
    ppl_absolute_difference = abs(hf_ppl - tvm_ppl)
    ppl_relative_difference = 100.0 * ppl_absolute_difference / hf_ppl
    average_logits_mae = total_abs_logit_error / compared_logit_values
    token_match_rate = 100.0 * matched_tokens / total_scored_tokens
    total_tvm_time = math.fsum(tvm_forward_times)
    average_tvm_time = total_tvm_time / len(tvm_forward_times)
    average_time_per_scored_token = total_tvm_time / total_scored_tokens

    passed = (
        np.isfinite(hf_ppl)
        and np.isfinite(tvm_ppl)
        and average_logits_mae <= args.mae_threshold
        and ppl_relative_difference <= args.relative_ppl_threshold
        and token_match_rate >= args.token_match_threshold
    )

    print("\n" + "=" * 60)
    print("WikiText-2 FP32 Equivalence Test")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Evaluation tokens: {len(evaluation_ids)}")
    print(f"Window size: {args.sequence_length}")
    print(f"Stride: {args.stride}")
    print(f"Number of windows: {len(windows)}")
    print(f"Scored tokens: {total_scored_tokens}")
    print("\nHF FP32:")
    print(f"  Perplexity: {hf_ppl:.6f}")
    print("\nTVM FP32:")
    print(f"  Perplexity: {tvm_ppl:.6f}")
    print("\nEquivalence:")
    print(f"  PPL Absolute Difference: {ppl_absolute_difference:.6f}")
    print(f"  PPL Relative Difference: {ppl_relative_difference:.6f} %")
    print(f"  Average Logits MAE: {average_logits_mae:.6e}")
    print(f"  Maximum Logits MAE: {maximum_window_mae:.6e}")
    print(f"  Maximum Absolute Logit Error: {maximum_absolute_logit_error:.6e}")
    print(
        f"  Token Match Rate: {token_match_rate:.2f} % "
        f"({matched_tokens}/{total_scored_tokens})"
    )
    print("\nTVM Performance:")
    print(f"  Warm-up windows: {args.warmup_windows}")
    print(f"  Average Forward Time / Window: {average_tvm_time:.6f} s")
    print(f"  Total Model Execution Time: {total_tvm_time:.6f} s")
    print(f"  Average Time / Scored Token: {average_time_per_scored_token:.6f} s")
    print(f"  End-to-end Equivalence Evaluation Time: {end_to_end_evaluation_time:.6f} s")
    print("\nConfigured PASS thresholds:")
    print(f"  Average logits MAE <= {args.mae_threshold:g}")
    print(f"  Relative PPL difference <= {args.relative_ppl_threshold:g} %")
    print(f"  Token match rate >= {args.token_match_threshold:g} %")
    print("\nResult:")
    print(f"  {'PASS' if passed else 'WARNING'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPT-2 Small WikiText-2 FP32 sliding-window equivalence benchmark"
    )
    parser.add_argument("--model", required=True, help="Compiled TVM FP32 shared library")
    parser.add_argument(
        "--dataset-path",
        default="./wikitext-2-raw-v1/test",
        help="WikiText-2 raw test file or a directory containing it",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer repository (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--hf-model",
        default=DEFAULT_HF_MODEL,
        help=f"HF PyTorch reference model (default: {DEFAULT_HF_MODEL})",
    )
    parser.add_argument("--sequence-length", type=positive_int, default=256)
    parser.add_argument("--stride", type=positive_int, default=128)
    parser.add_argument("--evaluation-tokens", type=int, default=1024, help="Number of WikiText-2 tokens to evaluate; 0 means the complete test set",)
    parser.add_argument(
        "--warmup-windows",
        type=positive_int,
        default=1,
        help="Number of untimed TVM warm-up calls (default: 1)",
    )
    parser.add_argument(
        "--num-threads",
        type=positive_int,
        default=None,
        help="Set TVM_NUM_THREADS before loading the compiled model",
    )
    parser.add_argument(
        "--mae-threshold",
        type=nonnegative_float,
        default=1e-3,
        help="PASS threshold for aggregate logits MAE (default: 1e-3)",
    )
    parser.add_argument(
        "--relative-ppl-threshold",
        type=nonnegative_float,
        default=0.1,
        help="PASS threshold for relative PPL difference in percent (default: 0.1)",
    )
    parser.add_argument(
        "--token-match-threshold",
        type=percentage,
        default=99.9,
        help="PASS threshold for argmax token match rate in percent (default: 99.9)",
    )
    args = parser.parse_args()
    if args.stride > args.sequence_length:
        parser.error("--stride must not exceed --sequence-length")
    if args.evaluation_tokens < 0:
        parser.error("--evaluation-tokens must be >= 0")
    if (args.evaluation_tokens != 0 and args.evaluation_tokens < args.sequence_length):
        parser.error(
            "--evaluation-tokens must be at least --sequence-length, "
            "or 0 for the complete test set"
        )
    if args.sequence_length > 1024:
        parser.error("GPT-2 supports at most 1024 positions")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
