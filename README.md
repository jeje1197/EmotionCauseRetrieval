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

The selected files retain complete conversations and their annotations; the
analysis includes every turn, not just turns selected for an experiment.
It prints and writes `data/output/reccon_dataset_analysis_valid/report.json` with
combined conversation and message totals, message-count distributions, evidence
status and cause-entry counts, per-conversation annotated-message counts, first
annotated-message and cause-message positions, shared and unique cause counts,
annotated-message counts by emotion, and a pre-embedding eligibility funnel.
It also writes a human-readable `report.md` in the analysis directory.
Use `--markdown-report` to choose another path. To include counts from an
existing saved run in the comparison, pass `--saved-results-dir` containing
`results_<split>_100_0.json` for each split. The saved results are post-embedding
counts, not pre-embedding eligibility counts. For example:

```bash
python -m src.scripts.dataset_analysis --saved-results-dir data/output --markdown-report DATASET_ANALYSIS_REPORT.md
```

The eligibility funnel counts conversations with any valid message at turn 5
or later, then those with one cause entry, a strictly earlier cause turn, and a
non-empty cause utterance. This is a proposed filter only; it does not change
the current selector. The report compares those counts to the selector's
current behavior. Distributions include count, minimum,
maximum, mean, median, and all tied modes. Per-conversation count distributions
include zeros. First-position distributions exclude conversations where no
such position exists, and the report counts those exclusions.

The report also estimates unique message turns to embed for all messages in
eligible conversations, through the last qualifying target, and through the
first qualifying target. These counts include target messages and all prior
retrieval candidates, assume embeddings are reused, and do not subtract existing
embeddings or count separate VAD-scoring calls. The JSON, written report, and
charts include the same dataset metrics for both complete eligible
conversations and first-target prefixes (every turn through the earliest
qualifying target). These post-filter scopes include all messages in the scope,
not only eligible targets. Prefix evidence is validated against the original
full conversation: valid references beyond the prefix still count for evidence
and sharing, but first-cause positions include only messages within the prefix.
The count of valid references outside the scope is reported.

Evidence fields are excluded from all cause analyses if they are not lists or
contain any entry that is not an integer turn number in the conversation.
This includes `"b"` and mixed valid/invalid lists; the whole field is excluded,
not just its invalid entries. The report separately counts excluded fields,
invalid entries in lists, fields containing both valid and invalid entries,
and exclusion reasons. Excluded fields are not counted as empty or missing.
Messages with excluded evidence remain in the conversation and message totals.
Valid evidence counts use list entries, including repeated turn references.
Missing and empty evidence fields both count as empty; missing fields are also
reported separately. Positions are one-based turn numbers. A cause position for
another message must reference a different turn; self-references are excluded.

For shared and unique counts, messages share a cause if their evidence lists
reference at least one common numbered turn; duplicates in one message do not
make it shared. A message with no shared cause is unique. The report includes both
all-conversation distributions and distributions restricted to conversations
with at least two annotated messages; the latter also include zero counts.
Emotion labels are trimmed and lowercased; `angry`, `happines`, `happy`, `sad`,
and `surprised` normalize to `anger`, `happiness`, `happiness`, `sadness`, and
`surprise`, respectively. `excited` remains separate.

The original histograms remain at `data/output/conversation_length_histogram.png`,
`data/output/evidence_message_histogram.png`, and
`data/output/evidence_cause_histogram.png`. Additional count, position, evidence
status, and emotion charts are saved beside the detailed report. The two
post-filter scopes have matching charts in `eligible_conversations` and
`first_target_prefixes` subdirectories of the analysis directory. Use `--data-dir`
to select a directory containing the three `reccon_selected_<split>.json`
files, `--analysis-dir` for the detailed report and additional charts, and
`--output`, `--evidence-output`, or `--cause-output` to change the original
histogram paths.

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
