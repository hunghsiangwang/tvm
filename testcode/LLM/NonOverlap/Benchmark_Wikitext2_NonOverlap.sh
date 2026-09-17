#!/usr/bin/env bash
# Compile two configurations and immediately compare them on the same fixed
# WikiText-2 subset. Posit modes are only run when explicitly selected.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TVM_ROOT="$(cd "${LLM_DIR}/../.." && pwd)"
PYTHON_BIN="${TVM_ROOT}/.venv/bin/python"
export PYTHONPATH="${TVM_ROOT}/python:${TVM_ROOT}/.local/python:${LLM_DIR}:${LLM_DIR}/.."
export TVM_LIBRARY_PATH="${TVM_ROOT}/build"
export LD_LIBRARY_PATH="${TVM_ROOT}/build/lib:${LD_LIBRARY_PATH:-}"
cd "${LLM_DIR}"

usage() {
  cat <<'EOF'
Usage:
  ./NonOverlap/Benchmark_Wikitext2_NonOverlap.sh prepare
  ./NonOverlap/Benchmark_Wikitext2_NonOverlap.sh CONFIG_A CONFIG_B [NUM_WINDOWS]

Examples:
  ./NonOverlap/Benchmark_Wikitext2_NonOverlap.sh fp32 fp16 8
  REUSE_EXISTING=true ./NonOverlap/Benchmark_Wikitext2_NonOverlap.sh hf fp32 8
  ./NonOverlap/Benchmark_Wikitext2_NonOverlap.sh fp32 posit32-es2 8
  ./NonOverlap/Benchmark_Wikitext2_NonOverlap.sh fp32 posit8-es1-mixed-quire 2
  ./NonOverlap/Benchmark_Wikitext2_NonOverlap.sh posit16-es1-mixed posit16-es1-mixed-quire 8

Configurations:
  hf (PyTorch FP32 reference; no compilation)
  fp32
  fp16
  int8-weights
  posit32-es1, posit32-es2
  posit16-es1, posit16-es2
  posit8-es1, posit8-es2
  posit16-es1-mixed, posit16-es2-mixed
  posit16-es1-mixed-quire, posit16-es2-mixed-quire
  posit8-es1-mixed, posit8-es2-mixed
  posit8-es1-mixed-quire, posit8-es2-mixed-quire

Environment overrides:
  REUSE_EXISTING=true   Reuse an existing .so instead of recompiling it.
  NUM_THREADS=N         Set TVM_NUM_THREADS for inference.
  USE_VECTORIZE=true    Pass --use-vectorize true during compilation.
EOF
}

# Sets the globals consumed by compilation and inference.
configure() {
  local name="$1"
  ONNX_PATH="./model/model.onnx"
  INPUT_DTYPE="float32"
  TARGET_DTYPE="float32"
  MIXED_DTYPE=""
  ACC_DTYPE=""
  MIXED_PRECISION=false
  USE_QUIRE=false
  DESCRIPTION="${name}"

  case "${name}" in
    hf)
      MODEL_PATH="hf"
      DESCRIPTION="Hugging Face / PyTorch FP32 reference"
      ;;
    fp32)
      MODEL_PATH="./model/GPT2_wt2_fp32_seq512.so"
      ;;
    fp16)
      ONNX_PATH="./gpt2-ONNX/onnx/model_fp16.onnx"
      MODEL_PATH="./model/GPT2_wt2_fp16_seq512.so"
      DESCRIPTION="FP16 ONNX weights/internal graph; float32 VM tensors"
      ;;
    int8-weights)
      ONNX_PATH="./gpt2-ONNX/onnx/model_int8.onnx"
      MODEL_PATH="./model/GPT2_wt2_int8_weights_seq512.so"
      DESCRIPTION="INT8 quantized weights; float32 VM tensors"
      ;;
    posit32-es1|posit32-es2)
      local es="${name##*-es}"
      TARGET_DTYPE="custom[posites${es}]32"
      MODEL_PATH="./model/GPT2_wt2_posit32_es${es}_seq512.so"
      ;;
    posit16-es1|posit16-es2|posit8-es1|posit8-es2)
      local bits="${name#posit}"
      bits="${bits%%-*}"
      local es="${name##*-es}"
      TARGET_DTYPE="custom[posites${es}]${bits}"
      MODEL_PATH="./model/GPT2_wt2_posit${bits}_es${es}_seq512.so"
      ;;
    posit16-es1-mixed|posit16-es2-mixed|posit16-es1-mixed-quire|posit16-es2-mixed-quire|\
    posit8-es1-mixed|posit8-es2-mixed|posit8-es1-mixed-quire|posit8-es2-mixed-quire)
      local bits="${name#posit}"
      bits="${bits%%-*}"
      local es="${name#*-es}"
      es="${es%%-*}"
      MIXED_PRECISION=true
      TARGET_DTYPE="custom[posites${es}]32"
      MIXED_DTYPE="custom[posites${es}]${bits}"
      ACC_DTYPE="custom[posites${es}]32"
      local suffix="mixed"
      if [[ "${name}" == *-quire ]]; then
        USE_QUIRE=true
        suffix="mixed_quire"
      fi
      MODEL_PATH="./model/GPT2_wt2_posit${bits}_es${es}_${suffix}_seq512.so"
      ;;
    *)
      echo "Unknown configuration: ${name}" >&2
      usage >&2
      exit 2
      ;;
  esac
  RUNTIME_DTYPE="${TARGET_DTYPE}"
}

compile_config() {
  local name="$1"
  configure "${name}"
  if [[ "${name}" == "hf" ]]; then
    return
  fi
  if [[ "${REUSE_EXISTING:-false}" == "true" && -f "${MODEL_PATH}" ]]; then
    echo "Reusing ${MODEL_PATH}"
    return
  fi

  local args=(
    --onnx-path "${ONNX_PATH}"
    --sequence-length 512
    --input-dtype "${INPUT_DTYPE}"
    --target-dtype "${TARGET_DTYPE}"
    --use-vectorize "${USE_VECTORIZE:-false}"
    --use-mixed-precision "${MIXED_PRECISION}"
    --use-quire "${USE_QUIRE}"
    --output "${MODEL_PATH}"
  )
  if [[ "${MIXED_PRECISION}" == "true" ]]; then
    args+=(
      --mixed-precision-dtype "${MIXED_DTYPE}"
      --mixed-precision-acc-dtype "${ACC_DTYPE}"
    )
  fi
  echo "Compiling ${name} -> ${MODEL_PATH}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/Compile_Wikitext2_NonOverlap.py" "${args[@]}"
}

if [[ "${1:-}" == "prepare" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/Inference_Wikitext2_NonOverlap.py" --prepare-only
  exit 0
fi
if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

CONFIG_A="$1"
CONFIG_B="$2"
NUM_WINDOWS="${3:-8}"
if ! [[ "${NUM_WINDOWS}" =~ ^[1-8]$ ]]; then
  echo "NUM_WINDOWS must be an integer from 1 to 8" >&2
  exit 2
fi

compile_config "${CONFIG_A}"
configure "${CONFIG_A}"
MODEL_A="${MODEL_PATH}"
DTYPE_A="${RUNTIME_DTYPE}"
MIXED_A="${MIXED_PRECISION}"
QUIRE_A="${USE_QUIRE}"
DESCRIPTION_A="${DESCRIPTION}"

compile_config "${CONFIG_B}"
configure "${CONFIG_B}"
MODEL_B="${MODEL_PATH}"
DTYPE_B="${RUNTIME_DTYPE}"
MIXED_B="${MIXED_PRECISION}"
QUIRE_B="${USE_QUIRE}"
DESCRIPTION_B="${DESCRIPTION}"

inference_args=(
  --model-a "${MODEL_A}"
  --dtype-a "${DTYPE_A}"
  --description-a "${DESCRIPTION_A}"
  --model-b "${MODEL_B}"
  --dtype-b "${DTYPE_B}"
  --description-b "${DESCRIPTION_B}"
  --windows-file ./wt2_windows_L512_seed0.npy
  --num-windows "${NUM_WINDOWS}"
)
if [[ "${MIXED_A}" == "true" ]]; then inference_args+=(--model-a-mixed-precision); fi
if [[ "${QUIRE_A}" == "true" ]]; then inference_args+=(--model-a-quire); fi
if [[ "${MIXED_B}" == "true" ]]; then inference_args+=(--model-b-mixed-precision); fi
if [[ "${QUIRE_B}" == "true" ]]; then inference_args+=(--model-b-quire); fi
inference_args+=(--num-threads "${NUM_THREADS:-12}")

echo "Running ${CONFIG_A} vs ${CONFIG_B} on ${NUM_WINDOWS} fixed window(s)"
"${PYTHON_BIN}" "${SCRIPT_DIR}/Inference_Wikitext2_NonOverlap.py" "${inference_args[@]}"
