# WikiText-2 GPT-2 benchmarks

The two evaluation paths intentionally implement different protocols.

```text
LLM/
├── Sliding_Window/       # FP32/reference sliding-window validation
├── NonOverlap/           # fixed 512-token datatype benchmark and shell driver
├── wt2_windows_L512_seed0.npy
└── wt2_windows_L512_seed0.meta
```

- `Sliding_Window/Compile_Wikitext-2.py` and
  `Sliding_Window/Inference_Wikitext-2.py` use configurable overlapping
  windows.  Overlap supplies more context, and only newly covered targets are
  scored.  This is the FP32/reference sanity path and is closer to conventional
  WikiText-2 evaluation.
- `NonOverlap/Compile_Wikitext2_NonOverlap.py` and
  `NonOverlap/Inference_Wikitext2_NonOverlap.py` use eight fixed, independently
  evaluated 512-token windows.  This lower-cost, reproducible subset is the
  datatype comparison protocol.  Its PPL must not be directly compared with a
  sliding-window PPL.

The non-overlap input is built from the complete local `wikitext2_test.txt`
tokenized as one stream with the exact tokenizer in `gpt2-ONNX/`.  Complete
512-token blocks are formed without shuffling or overlap.  NumPy RNG seed 0
selects eight blocks without replacement, the indices are sorted, and both the
tokens (`wt2_windows_L512_seed0.npy`) and provenance metadata
(`wt2_windows_L512_seed0.meta`, including SHA-256 provenance) are saved.
Existing files are loaded by default; `--regenerate-windows` is the explicit
replacement operation.

The downloaded ONNX/tokenizer directory, raw WikiText-2 text, compiled shared
libraries, and execution logs are local assets and are intentionally not
tracked.  Place them at `gpt2-ONNX/`, `wikitext2_test.txt`, `model/`, and
`log/`, respectively.  The fixed `.npy` subset and its provenance metadata are
tracked so every datatype configuration uses exactly the same token IDs.

Each model receives `(1, 512)` input IDs, a `(1, 512)` attention mask,
positions `0..511`, and 24 empty `(1, 12, 0, 64)` caches.  Windows never share
KV cache.  Positions 0..510 of the logits are scored against input tokens
1..511, so eight windows contain `8 * 511 = 4088` scored targets.

The shell driver compiles both selected configurations and immediately runs
inference with the correct model paths, runtime dtypes, and mixed/quire flags:

```bash
cd testcode/LLM
./NonOverlap/Benchmark_Wikitext2_NonOverlap.sh fp32 posit16-es1-mixed 8
```

The third argument is the number of windows. Using `2` always evaluates the
first two entries of the fixed eight-window file and never resamples. Set
`REUSE_EXISTING=true` to skip compilation when the expected `.so` already
exists.

The reported TVM forward time is one complete 512-token prefill/teacher-forcing
call with an empty input KV cache.  It is not per-token autoregressive decode
latency. Input construction, output conversion, metrics, and printing are
outside the timed region.

The repository's INT8 ONNX is a quantized-weight graph whose external
activation/cache/logits contract remains float32. Select the `int8-weights`
configuration; the driver automatically supplies the float32 VM tensor dtype
and reports that the internal weights are INT8.

Posit8/16 quire configurations are available in two distinct forms.  Both use
`quire_mul` to accumulate unrounded products in `quire<bits, es>` and round
only once at the end of each dot product.  A name such as
`posit8-es2-quire` keeps the complete model and matmul outputs in Posit8, so
the final quire value is rounded back to Posit8.  A name such as
`posit8-es2-mixed-quire` retains Posit8 matmul inputs but converts the final
quire value to the mixed-precision Posit32 output.  These configurations
intentionally remain separate from each other and from non-quire Posit32
accumulation.

The `fp16` configuration similarly imports `gpt2-ONNX/onnx/model_fp16.onnx`
without applying `ChangeDatatype`: its weights/internal graph remain FP16,
while its externally declared KV-cache and logits tensors remain float32.
