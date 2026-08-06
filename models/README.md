# Model files

Model weights are intentionally excluded from Git. Place the following files in the local project directory before running benchmarks.

## Language-model GGUF files

Directory:

```text
models/qwen3-vl-4b-custom-quants/
```

Required filenames:

```text
Qwen3VL-4B-Instruct-Q8_0-self.gguf
Qwen3VL-4B-Instruct-Q6_K-self.gguf
Qwen3VL-4B-Instruct-Q5_K_M-self.gguf
Qwen3VL-4B-Instruct-Q4_K_M-self.gguf
Qwen3VL-4B-Instruct-Q3_K_M-self.gguf
```

## Multimodal projector

Directory:

```text
models/qwen3-vl-4b-instruct-gguf/
```

Required filename:

```text
mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf
```

## Record checksums

After downloading or converting the files, record their hashes:

```bash
sha256sum \
  models/qwen3-vl-4b-custom-quants/*.gguf \
  models/qwen3-vl-4b-instruct-gguf/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf
```

The benchmark runner writes the model and projector hashes into each run manifest unless `HASH_MODE=skip` is set.

Upstream model: `Qwen/Qwen3-VL-4B-Instruct`. Do not commit model weights to this repository.
