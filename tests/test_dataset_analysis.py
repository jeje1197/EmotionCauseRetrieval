import json
import tempfile
import unittest
from pathlib import Path

from src.scripts.dataset_analysis import (
    analyze_conversations,
    analyze_eligibility,
    analyze_filtered_scopes,
    distribution,
    load_saved_experiment,
    render_markdown,
)


class DatasetAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.conversations = [
            {
                "conversation_id": "a",
                "turns": [
                    {"turn": 1, "emotion": "joy", "expanded emotion cause evidence": [1]},
                    {"turn": 2, "emotion": "joy", "expanded emotion cause evidence": [1, 1]},
                    {"turn": 3, "emotion": "sadness", "expanded emotion cause evidence": [2, "b"]},
                    {"turn": 4, "emotion": "neutral"},
                ],
            },
            {
                "conversation_id": "b",
                "turns": [
                    {"turn": 1, "emotion": "fear", "expanded emotion cause evidence": ["b"]},
                    {"turn": 2, "emotion": "neutral", "expanded emotion cause evidence": []},
                    {"turn": 3, "emotion": "joy", "expanded emotion cause evidence": [3]},
                    {"turn": 4, "emotion": "joy", "expanded emotion cause evidence": [1]},
                ],
            },
            {"conversation_id": "c", "turns": [{"turn": 1, "emotion": "neutral"}]},
        ]

    def test_analysis_counts_and_distributions(self):
        report, series = analyze_conversations(self.conversations)
        self.assertEqual(report["total_conversations"], 3)
        self.assertEqual(report["total_messages"], 9)
        self.assertEqual(report["messages_with_nonempty_evidence"], 4)
        self.assertEqual(report["messages_with_empty_or_missing_evidence"], 3)
        self.assertEqual(report["messages_with_missing_evidence_field"], 2)
        self.assertEqual(report["messages_with_multiple_cause_entries"], 1)
        self.assertEqual(report["excluded_invalid_evidence_fields"], 2)
        self.assertEqual(report["excluded_invalid_list_entries"], 2)
        self.assertEqual(report["excluded_invalid_fields_with_valid_entries"], 1)
        self.assertEqual(report["excluded_invalid_reasons"], {"non_integer_entry": 2})
        self.assertEqual(report["repeated_cause_entries_within_messages"], 1)
        self.assertEqual(report["annotated_messages_by_emotion"], {"joy": 4})
        self.assertEqual(series["causes_per_annotated_message"], [1, 2, 1, 1])
        self.assertEqual(series["first_annotated_message_position"], [1, 3])
        self.assertEqual(series["first_cause_of_another_message_position"], [1, 1])
        self.assertEqual(series["shared_messages_per_conversation"], [2, 0, 0])
        self.assertEqual(series["unique_messages_per_conversation"], [0, 2, 0])
        self.assertEqual(series["valid_evidence_fields_per_conversation"], [2, 3, 0])
        self.assertEqual(report["conversations_without_annotated_messages"], 1)
        self.assertEqual(report["conversations_without_a_numbered_cause_of_another_message"], 1)
        eligible = report["shared_unique_in_conversations_with_multiple_annotated_messages"]
        self.assertEqual(eligible["shared"]["count"], 2)
        self.assertEqual(eligible["shared"]["mode"], [0, 2])

    def test_self_references_do_not_count_as_causes_of_another_turn(self):
        report, series = analyze_conversations([{
            "conversation_id": "self",
            "turns": [{"turn": 1, "emotion": "joy", "expanded emotion cause evidence": [1]}],
        }])
        self.assertEqual(series["first_cause_of_another_message_position"], [])
        self.assertEqual(report["conversations_without_a_numbered_cause_of_another_message"], 1)
        self.assertEqual(report["shared_unique_in_conversations_with_multiple_annotated_messages"]["unique"]["count"], 0)

    def test_invalid_fields_are_excluded_and_classified(self):
        conversations = [{
            "conversation_id": "invalid",
            "turns": [
                {"turn": 1, "emotion": "joy", "expanded emotion cause evidence": [1, 5, "b", True]},
                {"turn": 2, "emotion": "joy", "expanded emotion cause evidence": "1"},
                {"turn": 3, "emotion": "joy", "expanded emotion cause evidence": []},
                {"turn": 4, "emotion": "joy", "expanded emotion cause evidence": [[1]]},
            ],
        }]
        report, series = analyze_conversations(conversations)
        self.assertEqual(report["excluded_invalid_evidence_fields"], 3)
        self.assertEqual(report["excluded_invalid_list_entries"], 4)
        self.assertEqual(report["excluded_invalid_fields_with_valid_entries"], 1)
        self.assertEqual(report["excluded_invalid_reasons"], {
            "field_not_list": 1, "non_integer_entry": 3, "unknown_turn_number": 1,
        })
        self.assertEqual(report["messages_with_empty_or_missing_evidence"], 1)
        self.assertEqual(report["messages_with_nonempty_evidence"], 0)
        self.assertEqual(series["valid_evidence_fields_per_conversation"], [1])
        self.assertEqual(series["first_annotated_message_position"], [])

    def test_emotion_aliases_are_normalized(self):
        labels = [" Angry ", "HAPPINES", "happy", "sad", "surprised", "excited", "happiness"]
        report, _ = analyze_conversations([{
            "conversation_id": "labels",
            "turns": [
                {"turn": number, "emotion": label, "expanded emotion cause evidence": [number]}
                for number, label in enumerate(labels, 1)
            ],
        }])
        self.assertEqual(report["annotated_messages_by_emotion"], {
            "anger": 1, "excited": 1, "happiness": 3, "sadness": 1, "surprise": 1,
        })

    def test_empty_distribution(self):
        self.assertEqual(distribution([]), {
            "count": 0, "min": None, "max": None, "mean": None, "median": None, "mode": [],
        })

    def test_eligibility_uses_any_qualifying_message_and_requires_nonempty_cause(self):
        def conversation(name, cause_utterance, evidence_at_five, evidence_at_six):
            return {
                "dataset_split": "train", "conversation_id": name,
                "turns": [
                    {"turn": 1, "utterance": cause_utterance, "emotion": "joy"},
                    *[
                        {"turn": number, "utterance": "text", "emotion": "joy"}
                        for number in (2, 3, 4)
                    ],
                    {"turn": 5, "utterance": "text", "emotion": "joy",
                     "expanded emotion cause evidence": evidence_at_five},
                    {"turn": 6, "utterance": "text", "emotion": "joy",
                     "expanded emotion cause evidence": evidence_at_six},
                ],
            }

        conversations = [
            conversation("later", "cause", [5], [1]),
            conversation("empty-cause", "  ", [1], []),
            conversation("invalid", "cause", [1, "b"], []),
            conversation("multiple", "cause", [1, 2], []),
        ]
        result = analyze_eligibility(conversations)
        self.assertEqual(result["totals"], {
            "conversations": 4,
            "valid_annotation_after_turn_4": 3,
            "single_cause_after_turn_4": 2,
            "single_previous_cause_after_turn_4": 2,
            "nonempty_cause_utterance": 1,
            "qualifying_messages": 1,
            "selected_target_and_golden": 3,
            "selected_single_cause": 1,
            "embed_all_messages": 6,
            "embed_through_last_target": 6,
            "embed_through_first_target": 6,
            "first_targets": 1,
            "first_target_prior_candidates": 5,
        })
        self.assertEqual(result["first_valid_annotation_after_turn_4"]["count"], 3)
        self.assertEqual(result["conversations_without_valid_annotation_after_turn_4"], 1)
        report, _ = analyze_conversations(conversations)
        report["eligibility"] = result
        text = render_markdown(report)
        self.assertIn("| Cause utterance is non-empty | 1 |", text)
        self.assertIn("| **Total** | **1** | **1** |", text)
        self.assertIn("not a change to the pipeline selector", text)
        self.assertIn("No saved experiment results were supplied", text)

    def test_filtered_scopes_and_embedding_counts(self):
        turns = [
            {"turn": i, "utterance": f"text {i}", "emotion": "joy"}
            for i in range(1, 8)
        ]
        turns[1]["expanded emotion cause evidence"] = [7]
        turns[2]["expanded emotion cause evidence"] = [7]
        turns[4]["expanded emotion cause evidence"] = [1]
        turns[6]["expanded emotion cause evidence"] = [2]
        conversations = [
            {"dataset_split": "train", "conversation_id": "eligible", "turns": turns},
            {"dataset_split": "test", "conversation_id": "ineligible", "turns": [
                {"turn": 1, "utterance": "text", "emotion": "neutral"},
            ]},
        ]
        eligibility = analyze_eligibility(conversations)
        totals = eligibility["totals"]
        self.assertEqual(totals["qualifying_messages"], 2)
        self.assertEqual(totals["embed_all_messages"], 7)
        self.assertEqual(totals["embed_through_last_target"], 7)
        self.assertEqual(totals["embed_through_first_target"], 5)
        self.assertEqual(totals["first_targets"], 1)
        self.assertEqual(totals["first_target_prior_candidates"], 4)
        filtered, series = analyze_filtered_scopes(conversations)
        full = filtered["eligible_conversations"]
        prefix = filtered["first_target_prefixes"]
        self.assertEqual(full["total_messages"], 7)
        self.assertEqual(prefix["total_messages"], 5)
        self.assertEqual(prefix["messages_with_nonempty_evidence"], 3)
        self.assertEqual(prefix["excluded_invalid_evidence_fields"], 0)
        self.assertEqual(prefix["valid_cause_entries_outside_included_turns"], 2)
        self.assertEqual(series["first_target_prefixes"]["first_cause_of_another_message_position"], [1])
        self.assertEqual(series["first_target_prefixes"]["shared_messages_per_conversation"], [2])
        report, _ = analyze_conversations(conversations)
        report["eligibility"] = eligibility
        report["filtered_scopes"] = filtered
        text = render_markdown(report)
        self.assertIn("| Through first qualifying target | 5 | 0 | 0 | 5 |", text)
        self.assertIn("### First-target prefixes", text)
        self.assertIn("Valid cause entries outside the included turns: 2", text)

    def test_no_eligible_conversations(self):
        conversations = [{"dataset_split": "train", "conversation_id": "none", "turns": []}]
        filtered, series = analyze_filtered_scopes(conversations)
        self.assertEqual(filtered["eligible_conversations"]["total_conversations"], 0)
        self.assertEqual(filtered["first_target_prefixes"]["distributions"]["messages_per_conversation"]["count"], 0)
        self.assertEqual(series["first_target_prefixes"]["messages_per_conversation"], [])

    def test_saved_results_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            for split in ("train", "valid", "test"):
                result = {
                    "split": split, "metrics": {"evaluated_conversation_count": 1},
                    "conversation_results": [{"conversation_id": "id"}],
                }
                (path / f"results_{split}_100_0.json").write_text(json.dumps(result), encoding="utf-8")
            self.assertEqual(load_saved_experiment(path)["evaluated_conversations"], 3)
            result["metrics"]["evaluated_conversation_count"] = 2
            (path / "results_test_100_0.json").write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_saved_experiment(path)


if __name__ == "__main__":
    unittest.main()
