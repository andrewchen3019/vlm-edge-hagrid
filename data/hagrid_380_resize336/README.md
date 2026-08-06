# Dataset

The final benchmark uses a balanced 19-class subset derived from:

```text
cj-mills/hagrid-classification-512p-no-gesture-150k
```

The source dataset card lists the license as CC BY-SA 4.0. The retained benchmark subset contains 380 resized derivative images: 20 images for each of 19 classes.

## Final directory

```text
data/hagrid_380_resize336/
├── images/
├── metadata.jsonl
└── dataset_manifest.json
```

Each metadata row contains the class, deterministic class index, repository-relative image path, source path, source SHA-256 hash, output SHA-256 hash, resolution, and JPEG quality.

## Rebuild

```bash
python3 src/data/download_hagrid_380_duckdb.py \
  --out data/hagrid_380_resize336 \
  --per-class 20 \
  --max-side 336 \
  --quality 92 \
  --overwrite
```

The downloader orders rows by upstream image path before selecting the first 20 images in each class. Compare the resulting hashes with the committed metadata before treating a regenerated subset as identical to the final benchmark.
