# EmotionFactRetrieval

A small research pipeline for downloading, formatting, selecting, embedding, and evaluating the RECCON dataset.

## Setup

1. Create and activate your Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Pipeline stages

Run the full pipeline from the repository root using:

```bash
python -m src.pipeline --stage all
```

Run individual stages:

```bash
python -m src.pipeline --stage download
python -m src.pipeline --stage format
python -m src.pipeline --stage select
python -m src.pipeline --stage embed
python -m src.pipeline --stage cleanup
```

The cleanup stage removes raw downloads and intermediate formatted or VAD-only files while preserving `*_selected.json` and `*_embedded_semantic.json` outputs by default.

## Experiment

After `embed` has completed, choose either split names or a specific split file.

Run selected splits:

```bash
python -m src.scripts.run_experiment --splits train valid
```

Run all splits explicitly:

```bash
python -m src.scripts.run_experiment --splits all
```

To run for a specific split file:

```bash
python -m src.scripts.run_experiment --input data/reccon_train_embedded_semantic.json
```

Note: `--input` and `--splits` are mutually exclusive, and one is required.

To customize the semantic/emotion weight sweep:

```bash
python -m src.scripts.run_experiment --weights 1.0 0.75 0.5 0.25 0.0
```

To write outputs to a custom directory:

```bash
python -m src.scripts.run_experiment --output-dir data/output
```

Each result file is written as:

- `data/output/results_(train|valid|test)_(semantic_weight_percent)_(vad_weight_percent).json`

Examples:

- `data/output/results_train_75_25.json`
- `data/output/results_valid_50_50.json`

## Output files

- `data/reccon_selected.json`
- `data/reccon_embedded_semantic.json`
- `data/output/results_*.json`

Intermediate files such as `data/dailydialog_*.json`, `data/reccon_formatted*.json`, and `data/reccon*_embedded.json` are generated artifacts and can be removed with the cleanup stage.

