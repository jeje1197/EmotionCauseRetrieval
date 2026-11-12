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

## Dataset analysis

After the select stage, analyze all conversations in the train, valid, and test
selected split files:

```bash
python -m src.scripts.dataset_analysis
```

The script prints the total number of conversations and the mean, median, and
mode of their message counts. It also reports the total number of messages
containing the `expanded emotion cause evidence` field and the mean, median,
and mode of those message counts per conversation. A message is counted once
when the field is present, regardless of how many evidence entries it contains;
conversations without such messages count as zero. All modes are reported if
there is a tie.

The script also reports totals and per-conversation mean, median, and mode for
messages with non-empty evidence, exactly one cause, and multiple causes.
Causes are counted as entries in the evidence list; empty lists and missing
fields do not count as non-empty evidence. Each per-conversation statistic
includes conversations with zero matching messages.

The histograms are saved to `data/output/conversation_length_histogram.png`,
`data/output/evidence_message_histogram.png`, and
`data/output/evidence_cause_histogram.png`. The last plot compares the number of
messages per conversation with any annotated cause, exactly one cause, or two
or more causes. These categories overlap: the first includes the other two.
The bars count messages, not individual message-cause pairs. Use `--data-dir` to choose another directory
containing the three `reccon_selected_<split>.json` files, `--output` to change
the conversation-length histogram path, `--evidence-output` to change the
field-presence histogram path, and `--cause-output` to change the new plot path.

## VAD stability

After producing the selected split files, generate three independent VAD
responses for every message in every split:

```bash
python -m src.scripts.vad_stability
```

The script uses the same VAD model and prompt as the embedding stage, with
temperature 0 and seed 0. These settings request reproducible decoding but do
not guarantee identical Gemini responses. A full fresh run requires three API
requests per message and may incur substantial time and cost.

The originals are not modified. Separate resumable copies are saved as
`data/vad_stability/reccon_<split>_vad_stability.json`, with scores in
`vad_scores_1`, `vad_scores_2`, and `vad_scores_3`. Existing non-empty
numbered scores are preserved; only missing or empty scores are generated.
Use `--overwrite` to regenerate all three. Checkpoints are written every 50
scores by default; `--checkpoint-interval` changes this. After an interrupted
overwrite, repeat with `--overwrite` to replace all runs consistently.

The combined report is written to `data/vad_stability/vad_stability_report.json`.
It includes each run's per-dimension mean and population standard deviation,
plus per-dimension mean within-message standard deviation, mean and maximum
within-message range, and pairwise mean absolute differences. Comparisons use
only messages with all three scores; the report also records complete and
incomplete counts. Use `--report-only` to recalculate without API requests,
or `--input-dir`, `--output-dir`, and `--report` to change paths.

For evaluation, generate semantic embeddings from each stability copy. For
example, run the following for each of `train`, `valid`, and `test`, replacing
`train` throughout:

```bash
python -m src.scripts.embed_dataset --input data/vad_stability/reccon_train_vad_stability.json --output data/vad_stability/reccon_train_embedded_semantic.json --splits train --only-semantic
python -m src.scripts.run_experiment --input data/vad_stability/reccon_train_embedded_semantic.json
```

Evaluation reads `vad_scores_1`; newly formatted and generated datasets use
that field instead of `vad_scores`. Older embedded files with only the legacy
field must be regenerated or migrated deliberately before evaluation. The
stability script does not copy legacy scores into any numbered run.

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

To map cosine similarities into `[0, 1]` instead of `[-1, 1]`:

```bash
python -m src.scripts.run_experiment --splits all --cosine-zero-one
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
