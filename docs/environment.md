# Validated environment

Fill or regenerate this document from the Jetson with:

```bash
bash scripts/collect_environment.sh
```

Known final-project configuration:

- Hardware: NVIDIA Jetson Orin Nano Super Developer Kit, 8 GB unified memory
- Power mode: 15 W
- Input resolution: maximum side 336 pixels
- llama.cpp submodule commit: `9de0fcf2b3e587a43f293d9a2b6ec0a32991f768`
- Context size: 512
- GPU layers: all requested
- Fit: off
- Parallel slots: 1
- Batch size: 128
- Microbatch size: 64
- MTMD batch tokens: 128
- KV cache: Q8_0 for keys and values
- Flash attention: on

The generated snapshot should also include JetPack, Jetson Linux, CUDA, Python, installed Python packages, exact model hashes, and the repository commit used for a release.
