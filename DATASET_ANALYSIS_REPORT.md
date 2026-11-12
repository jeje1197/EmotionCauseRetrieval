# RECCON dataset analysis

Generated from the selected train, valid, and test files. All turns are
included in dataset statistics; eligibility is evaluated before embedding.

## Dataset overview

- Conversations: 1,106
- Messages: 11,104
- Messages with valid non-empty evidence: 5,466
- Messages with empty or missing evidence: 5,243
- Invalid evidence fields excluded: 395 (403 invalid list entries; 204 fields also contain valid entries).
- Valid annotated messages with multiple cause entries: 2,460

Invalid fields, including mixed valid/invalid lists, are excluded in full
from cause analyses. Messages remain in conversation-length totals. Cause
entry counts include duplicate references. Emotion labels are lowercased
and normalized using the aliases in the JSON report; excited stays separate.

## Distributions across all conversations

| Distribution | Count | Min | Max | Mean | Median | Mode |
|---|---:|---:|---:|---:|---:|---:|
| Messages per conversation | 1106 | 3 | 31 | 10.04 | 10.00 | 8 |
| Cause entries per annotated message | 5466 | 1 | 11 | 1.68 | 1.00 | 1 |
| Annotated messages per conversation | 1106 | 0 | 18 | 4.94 | 4.00 | 4 |
| First annotated message position | 1093 | 1 | 17 | 2.68 | 2 | 1 |
| First message causing another turn | 1068 | 1 | 16 | 2.72 | 2.00 | 1 |
| Shared-cause messages per eligible conversation | 1075 | 0 | 16 | 3.74 | 4 | 4 |
| Unique-cause messages per eligible conversation | 1075 | 0 | 7 | 1.33 | 1 | 0 |
| First valid annotated message after turn 4 | 958 | 5 | 21 | 6.17 | 5.00 | 5 |

First-annotation positions exclude 13 conversations without valid annotations.
First-cause positions exclude 38 conversations without a cause referencing another turn; self-references are excluded.
First annotations after turn 4 exclude 148 conversations.
Shared/unique distributions include only conversations with at least two valid
annotated messages. Messages share a cause if they reference the same numbered turn.

### Normalized emotions in valid annotated messages

| Emotion | Messages |
|---|---:|
| happiness | 4,039 |
| surprise | 468 |
| anger | 423 |
| sadness | 339 |
| disgust | 135 |
| fear | 59 |
| excited | 3 |

## Proposed pre-embedding eligibility criteria

A conversation qualifies if any message meets all of these rules:

1. The message is turn 5 or later.
2. Its evidence field is valid and contains exactly one cause entry.
3. That cause references a strictly earlier turn in the same conversation.
4. The cause turn has a non-empty, non-whitespace utterance.

| Cumulative criterion | Conversations |
|---|---:|
| All conversations | 1,106 |
| Valid, non-empty evidence after turn 4 | 958 |
| Exactly one valid cause entry | 725 |
| Cause is a strictly earlier turn | 557 |
| Cause utterance is non-empty | 557 |

| Split | Eligible conversations | Qualifying messages |
|---|---:|---:|
| train | 417 | 766 |
| valid | 20 | 28 |
| test | 120 | 185 |
| **Total** | **557** | **979** |

## Embedding estimates

Counts are unique message turns, embedded once and reused; they include
target queries and all earlier retrieval candidates. They do not reflect
already-computed embeddings, retries, or separate VAD-scoring calls.

| Scope | Train | Valid | Test | Total |
|---|---:|---:|---:|---:|
| All messages in eligible conversations | 4,706 | 235 | 1,453 | 6,394 |
| Through last qualifying target (all targets) | 4,031 | 186 | 1,227 | 5,444 |
| Through first qualifying target | 3,204 | 167 | 993 | 4,364 |
| First target queries only | 417 | 20 | 120 | 557 |
| Prior candidates for first targets | 2,787 | 147 | 873 | 3,807 |

## Distributions after filtering

Both scopes retain all messages in the stated scope, including non-target
messages and their annotations. They do not restrict statistics to qualifying targets.
The first-target prefix ends at the earliest qualifying target. Evidence is
validated against the original full conversation, so a valid reference outside
the prefix is not reclassified as invalid. Shared-cause counts use such references;
first-cause positions count only cause messages included in the scope.

### Complete eligible conversations

- Conversations: 557; messages: 6,394.
- Valid non-empty evidence: 3,102; empty or missing: 3,150; invalid excluded fields: 142.
- Invalid list entries: 142; invalid fields containing valid entries: 65.
- Valid messages with more than one cause entry: 1,093.
- Valid cause entries outside the included turns: 0.

| Distribution | Count | Min | Max | Mean | Median | Mode |
|---|---:|---:|---:|---:|---:|---:|
| Messages per conversation | 557 | 5 | 31 | 11.48 | 11 | 12 |
| Cause entries per annotated message | 3102 | 1 | 8 | 1.49 | 1.00 | 1 |
| Annotated messages per conversation | 557 | 1 | 18 | 5.57 | 5 | 4 |
| First annotated message position | 557 | 1 | 17 | 3.12 | 2 | 1 |
| First message causing another turn | 557 | 1 | 16 | 3.25 | 2 | 1 |
| Shared-cause messages per conversation | 557 | 0 | 16 | 3.92 | 4 | 2 |
| Unique-cause messages per conversation | 557 | 0 | 7 | 1.65 | 1 | 1 |
| Shared-cause messages, multiple annotations | 552 | 0 | 16 | 3.95 | 4.00 | 2 |
| Unique-cause messages, multiple annotations | 552 | 0 | 7 | 1.66 | 1.00 | 1 |

First-annotation positions exclude 0 conversations; first-cause positions exclude 0.
Shared/unique rows marked multiple annotations include only conversations with at least two valid annotated messages.

| Normalized emotion | Annotated messages |
|---|---:|
| happiness | 2,404 |
| surprise | 322 |
| sadness | 169 |
| anger | 143 |
| disgust | 48 |
| fear | 15 |
| excited | 1 |

### First-target prefixes

- Conversations: 557; messages: 4,364.
- Valid non-empty evidence: 2,013; empty or missing: 2,242; invalid excluded fields: 109.
- Invalid list entries: 109; invalid fields containing valid entries: 43.
- Valid messages with more than one cause entry: 588.
- Valid cause entries outside the included turns: 0.

| Distribution | Count | Min | Max | Mean | Median | Mode |
|---|---:|---:|---:|---:|---:|---:|
| Messages per conversation | 557 | 5 | 21 | 7.83 | 7 | 5 |
| Cause entries per annotated message | 2013 | 1 | 6 | 1.37 | 1 | 1 |
| Annotated messages per conversation | 557 | 1 | 13 | 3.61 | 3 | 3 |
| First annotated message position | 557 | 1 | 17 | 3.12 | 2 | 1 |
| First message causing another turn | 557 | 1 | 16 | 3.27 | 2 | 1 |
| Shared-cause messages per conversation | 557 | 0 | 12 | 2.44 | 2 | 2 |
| Unique-cause messages per conversation | 557 | 0 | 5 | 1.17 | 1 | 1 |
| Shared-cause messages, multiple annotations | 507 | 0 | 12 | 2.68 | 2 | 2 |
| Unique-cause messages, multiple annotations | 507 | 0 | 5 | 1.19 | 1 | 0 |

First-annotation positions exclude 0 conversations; first-cause positions exclude 0.
Shared/unique rows marked multiple annotations include only conversations with at least two valid annotated messages.

| Normalized emotion | Annotated messages |
|---|---:|
| happiness | 1,511 |
| surprise | 244 |
| sadness | 116 |
| anger | 96 |
| disgust | 33 |
| fear | 12 |
| excited | 1 |

## Comparison with the current selector

- Current selection yields 619 target/golden pairs
  and 390 with single-entry target evidence.
- The selector prefers the first single-cause target before checking whether
  it has a prior cause; it can miss a later qualifying message. If there are
  no single-cause candidates it falls back to multiple-cause targets.
- The proposed rules are an analysis, not a change to the pipeline selector.

## Supplied saved experiment

The supplied results report 99 evaluated conversations.
These are post-embedding results, not the pre-embedding eligibility count.
The historical experiment discussed in the project investigation used
an evaluator that required features for every turn even though embedding
stopped at the target. The current evaluator instead checks turns through
the target; saved counts alone do not establish which version was used.

| Split | Evaluated | Result file |
|---|---:|---|
| train | 75 | `data\output\results_train_100_0.json` |
| valid | 5 | `data\output\results_valid_100_0.json` |
| test | 19 | `data\output\results_test_100_0.json` |
