# Results

The final results are organized by artifact type:

```text
results/
├── image_results/     Per-image predictions and request timings
├── tegrastat_logs/    Jetson telemetry sampled at approximately 1 Hz
├── server_logs/       llama-server output for the corresponding run
├── manifests/         Model, prompt, dataset, commit, and configuration hashes
└── summary/           Cross-model statistics
```

A complete final run should have four files with the same stem:

```text
qwen3vl4b_q5_336px_380_fullgpu.csv
qwen3vl4b_q5_336px_380_fullgpu_tegrastats.txt
qwen3vl4b_q5_336px_380_fullgpu_server.log
qwen3vl4b_q5_336px_380_fullgpu_manifest.json
```

## Metric notes

- `wall_s` excludes local image reading, Base64 encoding, request construction, and client JSON serialization.
- Server latency is `(prompt_ms + predicted_ms) / 1000`.
- Board energy per image is estimated from average active-window `VDD_IN` multiplied by summed request wall time.
- Gesture-family accuracy is secondary to exact-label accuracy.
- The statistics compiler has manually validated active-window starts for the five retained final tegrastats logs and also provides an automatic window mode for new logs.

## Q8 server log check

Before publishing a release, verify that the retained Q8 server log corresponds to the final stable Q8 prediction CSV. An older Q8 log containing multi-second stalls must be archived rather than presented as the matching final run.
