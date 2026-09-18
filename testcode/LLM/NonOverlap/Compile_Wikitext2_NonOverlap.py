"""Compile the existing BYODT GPT-2 pipeline for fixed-length WT2 windows."""

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
from tvm.relax.frontend.change_dataype import ChangeDatatype
from tvm.relax.frontend.onnx import from_onnx

from Compile import bind_symbolic_vars_if_present, optimize_ir_module, str2bool
from gpt2_dtype_utils import (
    parse_custom_dtype,
    register_custom_datatypes,
    validate_dtype_arg,
)


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
    """Inspect the actual decoder-with-past interface before Relax import."""

    inputs = {value.name: value for value in model.graph.input}
    outputs = {value.name: value for value in model.graph.output}
    expected_order = ["input_ids"] + [
        f"past_key_values.{layer}.{kind}"
        for layer in range(NUM_LAYERS)
        for kind in ("key", "value")
    ] + ["attention_mask", "position_ids"]
    actual_order = [value.name for value in model.graph.input]
    if actual_order != expected_order:
        raise ValueError(
            "Unexpected ONNX input order.\n"
            f"Expected: {expected_order}\nActual: {actual_order}"
        )
    if "logits" not in outputs:
        raise ValueError("ONNX model does not expose a logits output")

    shapes = {name: value_info_shape(value) for name, value in inputs.items()}
    logits_shape = value_info_shape(outputs["logits"])
    cache_shape = shapes["past_key_values.0.key"]
    if shapes["input_ids"] != ["batch_size", "sequence_length"]:
        raise ValueError(f"Unexpected input_ids metadata: {shapes['input_ids']}")
    if shapes["position_ids"] != ["batch_size", "sequence_length"]:
        raise ValueError(f"Unexpected position_ids metadata: {shapes['position_ids']}")
    if cache_shape != ["batch_size", NUM_HEADS, "past_sequence_length", HEAD_DIM]:
        raise ValueError(f"Unexpected KV-cache metadata: {cache_shape}")
    if logits_shape != ["batch_size", "sequence_length", VOCAB_SIZE]:
        raise ValueError(f"Unexpected logits metadata: {logits_shape}")

    print("ONNX decoder-with-past interface:")
    print(f"  input_ids: {shapes['input_ids']}")
    print(f"  KV cache (x24): {cache_shape}")
    print(f"  attention_mask: {shapes['attention_mask']}")
    print(f"  position_ids: {shapes['position_ids']}")
    print(f"  logits: {logits_shape}")


def correct_decoder_sequence_metadata(model):
    """Express the full-current-sequence contract implemented by the graph."""

    attention_mask = next(
        value for value in model.graph.input if value.name == "attention_mask"
    )
    attention_mask.type.tensor_type.shape.dim[1].dim_param = (
        "past_sequence_length + sequence_length"
    )
    for value in model.graph.output:
        if value.name.startswith("present."):
            value.type.tensor_type.shape.dim[2].dim_param = (
                "past_sequence_length + sequence_length"
            )


def build_dtype_converter(input_dtype, target_dtype):
    def converter(mod):
        return ChangeDatatype(input_dtype, target_dtype)(mod)["main"]

    return converter


def compile_model(args):
    """Run the same Relax/BYODT/TIR pipeline as Compile.py at sequence length L."""

    model = onnx.load_model(args.onnx_path)
    inspect_and_validate_onnx(model)
    correct_decoder_sequence_metadata(model)

    mod = from_onnx(model)
    mod = transform.DecomposeOpsForInference()(mod)
    mod = bind_symbolic_vars_if_present(
        mod,
        {
            "batch_size": 1,
            "sequence_length": args.sequence_length,
            "past_sequence_length": 0,
        },
    )
    if args.target_dtype != args.input_dtype:
        main = build_dtype_converter(args.input_dtype, args.target_dtype)(mod)
        mod = tvm.IRModule({"main": main})
    if args.use_mixed_precision:
        mod = transform.ToMixedPrecisionCustom(
            args.target_dtype,
            args.mixed_precision_dtype,
            args.mixed_precision_acc_dtype,
        )(mod)
    mod = transform.LegalizeOps()(mod)
    with tvm.transform.PassContext(
        opt_level=0, config={"tirx.disable_vectorize": not args.use_vectorize}
    ):
        mod = transform.FoldConstant()(mod)
        mod = optimize_ir_module(mod, use_vectorize=args.use_vectorize)
        if args.use_quire:
            from tir_transform_matmul_to_quire import InjectQuireMatmulElem

            mod = InjectQuireMatmulElem()(mod)
    return mod


def validate_mixed_precision(parser, args):
    if not args.use_mixed_precision:
        if not args.use_quire:
            return
        parsed_target = parse_custom_dtype(args.target_dtype)
        if (
            parsed_target is None
            or not parsed_target[0].startswith("posites")
            or parsed_target[1] not in (8, 16)
        ):
            parser.error(
                "pure-datatype --use-quire requires a custom[posites<es>]8 or "
                "custom[posites<es>]16 target dtype"
            )
        return
    configured = {
        "--target-dtype": args.target_dtype,
        "--mixed-precision-dtype": args.mixed_precision_dtype,
        "--mixed-precision-acc-dtype": args.mixed_precision_acc_dtype,
    }
    missing = [name for name, dtype in configured.items() if dtype is None]
    if missing:
        parser.error("required with --use-mixed-precision: " + ", ".join(missing))
    parsed = {name: parse_custom_dtype(dtype) for name, dtype in configured.items()}
    invalid = [name for name, value in parsed.items() if value is None]
    if invalid:
        parser.error("custom[name]bits required for: " + ", ".join(invalid))
    if len({value[0] for value in parsed.values()}) != 1:
        parser.error("all mixed-precision dtypes must use the same custom datatype name")
    source_bits = parsed["--target-dtype"][1]
    compute_bits = parsed["--mixed-precision-dtype"][1]
    accumulator_bits = parsed["--mixed-precision-acc-dtype"][1]
    if compute_bits >= source_bits:
        parser.error("--mixed-precision-dtype must be narrower than --target-dtype")
    if accumulator_bits < compute_bits:
        parser.error("--mixed-precision-acc-dtype must be at least the compute width")
    if args.use_quire:
        if not parsed["--mixed-precision-dtype"][0].startswith("posites"):
            parser.error("Quire is only available for custom[posites<es>] datatypes")
        if (compute_bits, accumulator_bits) not in ((8, 32), (16, 32)):
            parser.error("Quire supports Posit8/16 inputs with a Posit32 output")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile BYODT GPT-2 for fixed non-overlapping WikiText-2 windows"
    )
    parser.add_argument("--onnx-path", default="./model/model.onnx")
    parser.add_argument("--sequence-length", type=positive_int, default=512)
    parser.add_argument("--target", default="llvm")
    parser.add_argument("--input-dtype", type=validate_dtype_arg, default="float32")
    parser.add_argument("--target-dtype", type=validate_dtype_arg, default="float32")
    parser.add_argument("--use-mixed-precision", type=str2bool, default=False)
    parser.add_argument("--mixed-precision-dtype", type=validate_dtype_arg, default=None)
    parser.add_argument(
        "--mixed-precision-acc-dtype", type=validate_dtype_arg, default=None
    )
    parser.add_argument("--use-quire", type=str2bool, default=False)
    parser.add_argument("--use-vectorize", type=str2bool, default=False)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.sequence_length > 1024:
        parser.error("GPT-2 supports at most 1024 positions")
    validate_mixed_precision(parser, args)
    if args.output is None:
        safe_dtype = args.target_dtype.replace("[", "_").replace("]", "_")
        args.output = f"./model/GPT2_wt2_{safe_dtype}_seq{args.sequence_length}.so"
    return args


def main():
    args = parse_args()
    target = tvm.target.Target(args.target)
    dtypes = [args.input_dtype, args.target_dtype]
    if args.use_mixed_precision:
        dtypes += [args.mixed_precision_dtype, args.mixed_precision_acc_dtype]
    register_custom_datatypes(dtypes, target=target.kind.name)

    print("WikiText-2 non-overlap compile configuration:")
    for name in (
        "onnx_path",
        "sequence_length",
        "input_dtype",
        "target_dtype",
        "use_mixed_precision",
        "mixed_precision_dtype",
        "mixed_precision_acc_dtype",
        "use_quire",
        "use_vectorize",
        "target",
        "output",
    ):
        print(f"  {name}={getattr(args, name)}")
    print("  batch_size=1")
    print("  past_sequence_length=0")

    start = time.perf_counter()
    mod = compile_model(args)
    print(f"Relax import/optimization time: {time.perf_counter() - start:.2f} seconds")
    start = time.perf_counter()
    lib = tvm.relax.build(mod, target=target)
    print(f"Build time: {time.perf_counter() - start:.2f} seconds")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lib.export_library(str(output))
    print(f"Shared library written to: {output.resolve()}")


if __name__ == "__main__":
    main()
