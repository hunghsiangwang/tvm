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

SEQUENCE_LENGTH=256

OUTPUT="./model/GPT2_fp32_wikitext_seq256.so"

"${PYTHON_BIN}" "${SCRIPT_DIR}/Compile_Wikitext-2.py" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --output "${OUTPUT}"
