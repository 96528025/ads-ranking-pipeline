# Runbook

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For editable package usage:

```bash
pip install -e .
```

## Synthetic Pipeline

Quick demo:

```bash
python scripts/run_demo.py
```

Full model comparison:

```bash
python scripts/run_experiments.py
```

Feature ablation:

```bash
python scripts/run_feature_ablation.py
```

## Big-Data Feature Engineering

Local Spark requires Java 17 on this machine:

```bash
JAVA_HOME=$(/usr/libexec/java_home -v 17) python scripts/spark_build_features.py
```

Outputs:

```text
data/synthetic/raw_ads_events/dt=2026-05-08/*.parquet
data/synthetic/processed_ads_events/dt=2026-05-08/*.parquet
data/synthetic/processed_sample.csv
```

## Inference Benchmark

```bash
python scripts/benchmark_inference.py
```

Output:

```text
outputs/benchmarks/
```

## Ali-CCP Sample

Create:

```text
data/aliccp/sample.csv
```

Required columns:

```text
click
conversion
```

Then run:

```bash
python scripts/run_aliccp_sample.py
```

## Output Policy

Generated files are intentionally ignored by Git:

```text
data/
outputs/
checkpoints/
```

