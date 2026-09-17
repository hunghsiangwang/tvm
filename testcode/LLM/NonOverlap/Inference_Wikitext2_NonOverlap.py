"""Fixed 8 x 512 WikiText-2 teacher-forcing datatype benchmark."""

import argparse
import gc
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LLM_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(LLM_DIR))

import numpy as np
import tvm
from transformers import AutoTokenizer

from gpt2_dtype_utils import (
    convert_float_array_to_runtime_tensor,
    convert_runtime_tensor_to_float_numpy,
    register_custom_datatypes,
    validate_dtype_arg,
)
from wikitext2_nonoverlap_utils import (
    HEAD_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    SUBSET_SIZE,
    VOCAB_SIZE,
    WINDOW_LENGTH,
    compare_logits,
    load_or_create_fixed_windows,
    print_dataset_summary,
    stable_nll_sum,
)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def safe_exp(value):
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.exp(value))


def synchronize(device):
    if hasattr(device, "sync"):
        device.sync()


def make_tvm_inputs(input_ids, dtype, device):
    """Match the inspected ONNX order; every window starts with empty caches."""

    tensors = [tvm.runtime.tensor(input_ids, device=device)]
    empty_cache = np.zeros((1, NUM_HEADS, 0, HEAD_DIM), dtype=np.float32)
    tensors.extend(
        convert_float_array_to_runtime_tensor(empty_cache, dtype, device)
        for _ in range(NUM_LAYERS * 2)
    )
    tensors.append(
        tvm.runtime.tensor(
            np.ones((1, WINDOW_LENGTH), dtype=np.int64), device=device
        )
    )
    tensors.append(
        tvm.runtime.tensor(
            np.arange(WINDOW_LENGTH, dtype=np.int64)[None, :], device=device
        )
    )
    return tensors


@dataclass
class Aggregate:
    nll: float = 0.0
    scored_tokens: int = 0
    window_ppl: list = field(default_factory=list)
    forward_times: list = field(default_factory=list)


class HFRunner:
    is_tvm = False

    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device="cpu", dtype=torch.float32)
        self.model.eval()
        config = self.model.config
        actual = (
            config.n_layer,
            config.n_head,
            config.n_embd // config.n_head,
            config.vocab_size,
        )
        expected = (NUM_LAYERS, NUM_HEADS, HEAD_DIM, VOCAB_SIZE)
        if actual != expected:
            raise ValueError(f"HF GPT-2 config {actual} does not match {expected}")

    def run(self, input_ids):
        attention_mask = np.ones((1, WINDOW_LENGTH), dtype=np.int64)
        position_ids = np.arange(WINDOW_LENGTH, dtype=np.int64)[None, :]
        with self.torch.inference_mode():
            output = self.model(
                input_ids=self.torch.from_numpy(input_ids),
                attention_mask=self.torch.from_numpy(attention_mask),
                position_ids=self.torch.from_numpy(position_ids),
                use_cache=False,
            )
        logits = output.logits.detach().cpu().float().numpy()
        del output
        return logits, None


class TVMRunner:
    is_tvm = True

    def __init__(self, model_path, dtype, device):
        self.dtype = dtype
        self.device = device
        lib = tvm.runtime.load_module(model_path)
        self.vm = tvm.relax.vm.VirtualMachine(lib, device)

    def run(self, input_ids):
        # Input/cache creation is deliberately outside the timed region.
        inputs = make_tvm_inputs(input_ids, self.dtype, self.device)
        synchronize(self.device)
        start = time.perf_counter()
        output = self.vm["main"](*inputs)
        synchronize(self.device)
        elapsed = time.perf_counter() - start
        # Output conversion is also deliberately outside model execution timing.
        logits = convert_runtime_tensor_to_float_numpy(
            output[0], self.dtype, self.device
        )
        del output, inputs
        return logits, elapsed


def make_runner(model_path, dtype, hf_model, device):
    if model_path.lower() == "hf":
        if dtype != "float32":
            raise ValueError("The HF reference runner must use dtype float32")
        return HFRunner(hf_model)
    return TVMRunner(model_path, dtype, device)


def validate_logits(logits, model_name):
    expected = (1, WINDOW_LENGTH, VOCAB_SIZE)
    if logits.shape != expected:
        raise ValueError(
            f"{model_name} logits shape is {logits.shape}, expected {expected}; "
            "compile the model with --sequence-length 512"
        )


def add_model_result(aggregate, scored_logits, labels, elapsed):
    nll = stable_nll_sum(scored_logits, labels)
    aggregate.nll += nll
    aggregate.scored_tokens += labels.size
    aggregate.window_ppl.append(safe_exp(nll / labels.size))
    if elapsed is not None:
        aggregate.forward_times.append(elapsed)
    return nll


def print_model_summary(name, aggregate):
    corpus_ppl = safe_exp(aggregate.nll / aggregate.scored_tokens)
    window_values = np.asarray(aggregate.window_ppl, dtype=np.float64)
    print(f"{name}:")
    print(f"  Corpus PPL: {corpus_ppl:.8f}")
    print(
        "  Window PPL: "
        f"{np.mean(window_values):.8f} ± {np.std(window_values, ddof=0):.8f}"
    )
    if aggregate.forward_times:
        timings = np.asarray(aggregate.forward_times, dtype=np.float64)
        total = math.fsum(aggregate.forward_times)
        print("  Per-window Forward Time: " + ", ".join(f"{x:.6f}s" for x in timings))
        print(f"  Average Forward Time: {np.mean(timings):.6f} s")
        print(f"  Total Model Execution Time: {total:.6f} s")
        print(
            "  Average Model Execution Time / Scored Token: "
            f"{total / aggregate.scored_tokens:.9f} s"
        )
    else:
        print("  TVM execution timing: N/A (HF reference execution is excluded)")
    return corpus_ppl


def evaluate(args):
    if args.num_threads is not None:
        os.environ["TVM_NUM_THREADS"] = str(args.num_threads)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    windows, metadata = load_or_create_fixed_windows(
        args.text,
        tokenizer,
        args.windows_file,
        regenerate=args.regenerate_windows,
    )

    print("=" * 60)
    print("WikiText-2 Non-Overlapping Datatype Benchmark")
    print("=" * 60)
    print_dataset_summary(metadata, args.windows_file, args.num_windows)
    if args.prepare_only:
        print("\nFixed benchmark input is ready; no model execution requested.")
        return

    if args.model_a is None or args.model_b is None:
        raise ValueError(
            "--model-a and --model-b are required unless --prepare-only is set"
        )

    device = tvm.cpu()
    register_custom_datatypes((args.dtype_a, args.dtype_b), target="llvm")
    runner_a = make_runner(args.model_a, args.dtype_a, args.hf_model, device)
    runner_b = make_runner(args.model_b, args.dtype_b, args.hf_model, device)
    print("\nModel A:")
    print(f"  Path/source: {args.model_a}")
    print(f"  DType: {args.dtype_a}")
    print(f"  Mixed Precision: {args.model_a_mixed_precision}")
    print(f"  Quire: {args.model_a_quire}")
    print(f"  Description: {args.description_a}")
    print("Model B:")
    print(f"  Path/source: {args.model_b}")
    print(f"  DType: {args.dtype_b}")
    print(f"  Mixed Precision: {args.model_b_mixed_precision}")
    print(f"  Quire: {args.model_b_quire}")
    print(f"  Description: {args.description_b}")

    if args.warmup_windows:
        warmup_ids = np.ascontiguousarray(windows[0:1])
        for runner in (runner_a, runner_b):
            if runner.is_tvm:
                for _ in range(args.warmup_windows):
                    warm_logits, _ = runner.run(warmup_ids)
                    del warm_logits

    aggregate_a = Aggregate()
    aggregate_b = Aggregate()
    mae_by_window = []
    total_absolute_error = 0.0
    total_logit_values = 0
    total_matches = 0

    for index in range(args.num_windows):
        print("\n" + "-" * 60)
        source_index = metadata["selected_indices"][index]
        print(
            f"Window {index + 1} / {args.num_windows} "
            f"(source index {source_index})"
        )
        print("-" * 60, flush=True)
        input_ids = np.ascontiguousarray(windows[index : index + 1])
        labels = input_ids[0, 1:]

        logits_a, elapsed_a = runner_a.run(input_ids)
        validate_logits(logits_a, "Model A")
        scored_a = logits_a[0, :-1, :]
        nll_a = add_model_result(aggregate_a, scored_a, labels, elapsed_a)

        logits_b, elapsed_b = runner_b.run(input_ids)
        validate_logits(logits_b, "Model B")
        scored_b = logits_b[0, :-1, :]
        nll_b = add_model_result(aggregate_b, scored_b, labels, elapsed_b)

        absolute_sum, value_count, matches = compare_logits(scored_a, scored_b)
        window_mae = absolute_sum / value_count
        total_absolute_error += absolute_sum
        total_logit_values += value_count
        total_matches += matches
        mae_by_window.append(window_mae)

        print(
            f"Model A Forward Time: {elapsed_a:.6f} s"
            if elapsed_a is not None
            else "Model A Forward Time: N/A (HF)"
        )
        print(
            f"Model B Forward Time: {elapsed_b:.6f} s"
            if elapsed_b is not None
            else "Model B Forward Time: N/A (HF)"
        )
        print(f"Window A PPL: {safe_exp(nll_a / labels.size):.8f}")
        print(f"Window B PPL: {safe_exp(nll_b / labels.size):.8f}")
        print(f"Logits MAE: {window_mae:.8e}")
        print(f"Token Match: {matches} / {labels.size}", flush=True)

        del input_ids, labels, logits_a, logits_b, scored_a, scored_b
        gc.collect()

    expected_scored = args.num_windows * (WINDOW_LENGTH - 1)
    if (
        aggregate_a.scored_tokens != expected_scored
        or aggregate_b.scored_tokens != expected_scored
    ):
        raise AssertionError("Unexpected scored-token count")

    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    ppl_a = print_model_summary("Model A", aggregate_a)
    ppl_b = print_model_summary("Model B", aggregate_b)
    mae_values = np.asarray(mae_by_window, dtype=np.float64)
    ppl_difference = abs(ppl_a - ppl_b)
    relative = 100.0 * ppl_difference / ppl_a
    print("Comparison:")
    print(f"  PPL Difference: {ppl_difference:.8f}")
    print(f"  Relative PPL Difference: {relative:.8f} %")
    print(f"  Overall Logits MAE: {total_absolute_error / total_logit_values:.8e}")
    print("  Per-window Logits MAE: " + ", ".join(f"{x:.8e}" for x in mae_values))
    print(
        "  Per-window Logits MAE Mean ± Std: "
        f"{np.mean(mae_values):.8e} ± {np.std(mae_values, ddof=0):.8e}"
    )
    print(f"  Token Match: {total_matches} / {expected_scored}")
    print(f"  Token Match Rate: {100.0 * total_matches / expected_scored:.6f} %")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two models on the fixed WT2 8 x 512 non-overlap subset"
    )
    parser.add_argument("--model-a", help="TVM .so path, or 'hf' for PyTorch FP32")
    parser.add_argument("--model-b", help="TVM .so path, or 'hf' for PyTorch FP32")
    parser.add_argument("--dtype-a", type=validate_dtype_arg, default="float32")
    parser.add_argument("--dtype-b", type=validate_dtype_arg, default="float32")
    parser.add_argument("--description-a", default="")
    parser.add_argument("--description-b", default="")
    parser.add_argument("--model-a-mixed-precision", action="store_true")
    parser.add_argument("--model-b-mixed-precision", action="store_true")
    parser.add_argument("--model-a-quire", action="store_true")
    parser.add_argument("--model-b-quire", action="store_true")
    parser.add_argument("--text", default=str(LLM_DIR / "wikitext2_test.txt"))
    parser.add_argument(
        "--tokenizer", default=str(LLM_DIR / "gpt2-ONNX"),
        help="Exact GPT-2 tokenizer directory/repository",
    )
    parser.add_argument("--hf-model", default="openai-community/gpt2")
    parser.add_argument(
        "--windows-file",
        default=str(LLM_DIR / "wt2_windows_L512_seed0.npy"),
    )
    parser.add_argument("--num-windows", type=positive_int, default=SUBSET_SIZE)
    parser.add_argument("--num-threads", type=positive_int, default=None)
    parser.add_argument("--warmup-windows", type=int, choices=range(0, 9), default=0)
    parser.add_argument("--regenerate-windows", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.num_windows > SUBSET_SIZE:
        parser.error(f"--num-windows must be between 1 and {SUBSET_SIZE}")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
