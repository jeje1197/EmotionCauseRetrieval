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
