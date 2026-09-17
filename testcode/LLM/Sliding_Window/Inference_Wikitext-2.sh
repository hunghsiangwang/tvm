#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TVM_ROOT="$(cd -- "${LLM_DIR}/../.." && pwd)"

cd "${LLM_DIR}"

PYTHON_BIN="${TVM_ROOT}/.venv/bin/python"
export PYTHONPATH="${TVM_ROOT}/python:${TVM_ROOT}/.local/python:${LLM_DIR}:${LLM_DIR}/.."
export TVM_LIBRARY_PATH="${TVM_ROOT}/build"
export LD_LIBRARY_PATH="${TVM_ROOT}/build/lib:${LD_LIBRARY_PATH:-}"

MODEL="./model/GPT2_fp32_wikitext_seq256.so"
DATASET_PATH="./wikitext2_test.txt"

SEQUENCE_LENGTH=256
STRIDE=128
EVALUATION_TOKENS=0
NUM_THREADS=12

"${PYTHON_BIN}" "${SCRIPT_DIR}/Inference_Wikitext-2.py" \
    --model "${MODEL}" \
    --dataset-path "${DATASET_PATH}" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --stride "${STRIDE}" \
    --evaluation-tokens "${EVALUATION_TOKENS}" \
    --num-threads "${NUM_THREADS}"
