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
```

## Experiment

After `embed` has completed, run the retrieval experiment with:

```bash
python -m src.scripts.run_experiment
```

To save results to a file:

```bash
python -m src.scripts.run_experiment --output data/experiment_results.json
```

To customize the semantic/emotion weight sweep:

```bash
python -m src.scripts.run_experiment --weights 1.0 0.75 0.5 0.25 0.0
```

## Output files

- `data/reccon_formatted.json`
- `data/reccon_selected.json`
- `data/reccon_embedded.json`

