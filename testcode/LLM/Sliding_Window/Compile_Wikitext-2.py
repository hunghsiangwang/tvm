import argparse
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LLM_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(LLM_DIR))

import onnx
import tvm
from tvm.relax import transform
from tvm.relax.frontend.onnx import from_onnx

from Compile import bind_symbolic_vars_if_present, optimize_ir_module, str2bool


os.chdir(LLM_DIR)

NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64
VOCAB_SIZE = 50257


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def value_info_shape(value_info):
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            dims.append(dim.dim_param or "?")
    return dims


def inspect_and_validate_onnx(model):
    """Validate the decoder-with-past interface used by the sequence evaluator."""

    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    required_inputs = {"input_ids", "attention_mask", "position_ids"}
    required_inputs.update(
        f"past_key_values.{layer}.{kind}"
        for layer in range(NUM_LAYERS)
        for kind in ("key", "value")
    )
    missing_inputs = sorted(required_inputs.difference(inputs))
    if missing_inputs:
        raise ValueError("ONNX model is missing required inputs: " + ", ".join(missing_inputs))
    if "logits" not in outputs:
        raise ValueError("ONNX model does not expose a 'logits' output")

    input_ids_shape = value_info_shape(inputs["input_ids"])
    attention_mask_shape = value_info_shape(inputs["attention_mask"])
    position_ids_shape = value_info_shape(inputs["position_ids"])
    logits_shape = value_info_shape(outputs["logits"])
    first_cache_shape = value_info_shape(inputs["past_key_values.0.key"])
    if len(input_ids_shape) != 2 or len(position_ids_shape) != 2:
        raise ValueError("input_ids and position_ids must both be rank-2 tensors")
    if len(attention_mask_shape) != 2:
        raise ValueError("attention_mask must be a rank-2 tensor")
    if first_cache_shape[1:] != [NUM_HEADS, "past_sequence_length", HEAD_DIM]:
        raise ValueError(f"Unexpected GPT-2 KV-cache shape: {first_cache_shape}")
    if len(logits_shape) != 3 or logits_shape[-1] != VOCAB_SIZE:
        raise ValueError(f"Unexpected GPT-2 logits shape: {logits_shape}")

    print("ONNX decoder-with-past interface:")
    print(f"  input_ids: {input_ids_shape}")
    print(f"  KV cache (x{NUM_LAYERS * 2}): {first_cache_shape}")
    print(f"  attention_mask metadata: {attention_mask_shape}")
    print(f"  position_ids: {position_ids_shape}")
    print(f"  logits: {logits_shape}")
    print(
        "  Sequence evaluation uses past_sequence_length=0 and an attention mask "
        "whose runtime length equals sequence_length."
    )


def correct_decoder_sequence_metadata(model):
    """Correct decode-step metadata while leaving the ONNX graph unchanged."""

    # Optimum exported this graph while tracing one current token, so the
    # attention/present annotations say past + 1.  Concat and attention nodes in
    # the graph use the full current sequence and therefore implement past +
    # sequence_length.  Relax consumes these annotations as input types, so make
    # that existing graph contract explicit before symbolic import.
    attention_mask = next(value for value in model.graph.input if value.name == "attention_mask")
    attention_mask.type.tensor_type.shape.dim[1].dim_param = (
        "past_sequence_length + sequence_length"
    )
    for value in model.graph.output:
        if value.name.startswith("present."):
            value.type.tensor_type.shape.dim[2].dim_param = (
                "past_sequence_length + sequence_length"
            )


def compile_model(onnx_path, sequence_length, use_vectorize=False):
    """Import GPT-2 as a static full-sequence FP32 Relax module."""

    model = onnx.load_model(onnx_path)
    inspect_and_validate_onnx(model)

    correct_decoder_sequence_metadata(model)
    mod = from_onnx(model)
    mod = transform.DecomposeOpsForInference()(mod)
    mod = bind_symbolic_vars_if_present(
        mod,
        {
            "batch_size": 1,
            "sequence_length": sequence_length,
            "past_sequence_length": 0,
        },
    )
    mod = tvm.relax.transform.LegalizeOps()(mod)
    with tvm.transform.PassContext(
        opt_level=0, config={"tirx.disable_vectorize": not use_vectorize}
    ):
        mod = tvm.relax.transform.FoldConstant()(mod)
        mod = optimize_ir_module(mod, use_vectorize=use_vectorize)
    return mod


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile GPT-2 ONNX for FP32 WikiText-2 full-sequence evaluation"
    )
    parser.add_argument(
        "--onnx-path",
        default="./model/model.onnx",
        help="Path to the onnx-community/gpt2-ONNX model",
    )
    parser.add_argument(
        "--sequence-length",
        type=positive_int,
        default=256,
        help="Static full-sequence length (default: 256)",
    )
    parser.add_argument(
        "--target",
        default="llvm",
        help="TVM target name or JSON configuration (default: llvm)",
    )
    parser.add_argument(
        "--use-vectorize",
        type=str2bool,
        default=False,
        help="Enable the existing TIR vectorization optimization (true/false)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output shared library path (default includes --sequence-length)",
    )
    args = parser.parse_args()
    if args.sequence_length > 1024:
        parser.error("GPT-2 supports at most 1024 positions")
    if args.output is None:
        args.output = f"./model/GPT2_fp32_wikitext_seq{args.sequence_length}.so"
    return args


def main():
    args = parse_args()
    target = tvm.target.Target(args.target)

    print("WikiText-2 FP32 compile configuration:")
    print(f"  onnx_path={args.onnx_path}")
    print("  batch_size=1")
    print(f"  sequence_length={args.sequence_length}")
    print("  past_sequence_length=0")
    print("  input_dtype=float32")
    print("  target_dtype=float32")
    print(f"  target={args.target}")
    print(f"  use_vectorize={args.use_vectorize}")
    print(f"  output={args.output}")

    compile_start = time.perf_counter()
    mod = compile_model(args.onnx_path, args.sequence_length, args.use_vectorize)
    compile_time = time.perf_counter() - compile_start
    print(f"Relax import/optimization time: {compile_time:.2f} seconds")

    build_start = time.perf_counter()
    lib = tvm.relax.build(mod, target=target)
    build_time = time.perf_counter() - build_start
    print(f"Build time: {build_time:.2f} seconds")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_start = time.perf_counter()
    lib.export_library(str(output_path))
    export_time = time.perf_counter() - export_start
    print(f"Export time: {export_time:.2f} seconds")
    print(f"Shared library written to: {output_path}")


if __name__ == "__main__":
    main()
