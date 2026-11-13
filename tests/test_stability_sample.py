import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from src.scripts.stability_sample import (
    generate_scores,
    sample_conversations,
    validate_sample,
    write_report,
)
from src.scripts.test_vad_call import main as run_vad_smoke


def conversation(identifier, split="train", emotion="happiness"):
    turns = [{"turn": i, "utterance": f"message {i}", "emotion": emotion} for i in range(1, 8)]
    turns[4]["expanded emotion cause evidence"] = [5]
    turns[5]["expanded emotion cause evidence"] = [1]
    return {
        "conversation_id": identifier, "dataset_split": split,
        "target_turn": 5, "golden_turn": -1, "turns": turns,
    }


class StabilitySampleTests(unittest.TestCase):
    def test_stratified_deterministic_sample_and_target(self):
        population = [
            conversation(f"train-{i}", emotion="happy") for i in range(80)
        ] + [
            conversation(f"test-{i}", split="test", emotion="anger") for i in range(20)
        ]
        original = copy.deepcopy(population)
        first = sample_conversations(population)
        second = sample_conversations(list(reversed(population)))
        self.assertEqual(first, second)
        self.assertEqual(population, original)
        self.assertEqual(len(first), 10)
        self.assertEqual(sum(c["dataset_split"] == "train" for c in first), 8)
        self.assertEqual(sum(c["dataset_split"] == "test" for c in first), 2)
        self.assertTrue(all(c["target_turn"] == 6 and c["golden_turn"] == 1 for c in first))
        validate_sample(first, Path("sample.json"))

    def test_scoring_resume_and_stability_report(self):
        sample = sample_conversations([conversation(str(i)) for i in range(10)])
        score = {"valence": 0.1, "arousal": 0.2, "dominance": -0.3}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output, report_path = root / "sample.json", root / "embedded.json", root / "report.json"
            source.write_text(json.dumps(sample), encoding="utf-8")
            source_before = source.read_bytes()
            with patch("src.scripts.stability_sample.GeminiClient"), patch(
                "src.scripts.stability_sample.score_vad", side_effect=[score, score, RuntimeError("offline")]
            ):
                with self.assertRaisesRegex(RuntimeError, "offline"):
                    generate_scores(source, output, checkpoint_interval=50, max_retries=0, concurrent_requests=1)
            partial = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(partial[0]["turns"][0]["vad_scores_1"], score)
            self.assertEqual(partial[0]["turns"][0]["vad_scores_2"], score)
            self.assertNotIn("vad_scores_3", partial[0]["turns"][0])
            with self.assertRaisesRegex(ValueError, "incomplete VAD"):
                write_report(output, report_path)
            self.assertFalse(report_path.exists())
            with patch("src.scripts.stability_sample.GeminiClient"), patch(
                "src.scripts.stability_sample.score_vad", return_value=score
            ) as scorer:
                generate_scores(source, output, checkpoint_interval=4, max_retries=0)
                self.assertEqual(scorer.call_count, 16)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(source.read_bytes(), source_before)
            self.assertNotIn("vad_scores_1", result[0]["turns"][6])
            report = write_report(output, report_path)
            self.assertEqual(report["total_turns"], 6)
            self.assertEqual(report["complete_turns"], 6)
            self.assertEqual(report["incomplete_turns"], 0)
            self.assertEqual(report["runs"]["vad_scores_1"]["dimensions"]["valence"]["mean"], 0.1)
            self.assertEqual(report["within_turn_comparisons"]["valence"]["mean_within_turn_population_stddev"], 0.0)
            with patch("src.scripts.stability_sample.GeminiClient") as client:
                generate_scores(source, output)
                client.assert_not_called()
            modified = json.loads(source.read_text(encoding="utf-8"))
            modified[0]["turns"][0]["utterance"] = "changed"
            source.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                generate_scores(source, output)

    def test_parallel_requests_and_failed_batch_preserve_successes(self):
        sample = sample_conversations([conversation(str(i)) for i in range(10)])
        score = {"valence": 0.1, "arousal": 0.2, "dominance": -0.3}
        barrier = threading.Barrier(3, timeout=5)
        lock = threading.Lock()
        calls = 0

        def fake_score(client, text):
            nonlocal calls
            with lock:
                calls += 1
                position = calls
            barrier.wait()
            if position == 1:
                raise RuntimeError("request failed")
            return score

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "sample.json", root / "embedded.json"
            source.write_text(json.dumps(sample), encoding="utf-8")
            with patch("src.scripts.stability_sample.GeminiClient"), patch(
                "src.scripts.stability_sample.score_vad", side_effect=fake_score
            ):
                with self.assertRaisesRegex(RuntimeError, "request failed"):
                    generate_scores(source, output, concurrent_requests=3, checkpoint_interval=50, max_retries=0)
            self.assertEqual(calls, 3)
            turn = json.loads(output.read_text(encoding="utf-8"))[0]["turns"][0]
            self.assertEqual(sum(field in turn for field in ("vad_scores_1", "vad_scores_2", "vad_scores_3")), 2)

    def test_rejects_nonpositive_concurrency(self):
        with self.assertRaisesRegex(ValueError, "concurrency"):
            generate_scores(Path("in.json"), Path("out.json"), concurrent_requests=0)

    def test_single_call_uses_same_scorer(self):
        score = {"valence": 0.1, "arousal": 0.2, "dominance": 0.3}
        with patch("src.scripts.test_vad_call.GeminiClient") as client, patch(
            "src.scripts.test_vad_call.score_vad", return_value=score
        ) as scorer:
            run_vad_smoke()
            scorer.assert_called_once()
            self.assertIs(scorer.call_args.args[0], client.return_value)
            self.assertTrue(scorer.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
