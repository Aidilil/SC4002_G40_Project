import json
import unittest
from pathlib import Path

import torch

from src.data_loader import get_dataloaders, parse_annot_utt
from src.metrics import compute_slot_micro_f1
from src.model import JointBiLSTMAttention


def _write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")


class Stage1BaselineTests(unittest.TestCase):
    def test_parse_annot_utt_alignment(self):
        tokens, tags = parse_annot_utt("set alarm for [time : five am]")
        self.assertEqual(tokens, ["set", "alarm", "for", "five", "am"])
        self.assertEqual(tags, ["O", "O", "O", "B-time", "I-time"])

    def test_slot_micro_f1_exact_match(self):
        gold = [[("time", "five am")], [("song", "hello")]]
        pred = [[("time", "five am")], [("song", "rolling")]]
        score = compute_slot_micro_f1(gold, pred)
        self.assertEqual(round(score, 2), 50.0)

    def test_model_contract_shapes_and_loss(self):
        model = JointBiLSTMAttention(
            vocab_size=100,
            embedding_dim=16,
            hidden_dim=8,
            num_intents=60,
            num_slots=111,
            num_layers=1,
            dropout=0.1,
        )
        input_ids = torch.randint(0, 100, (4, 7))
        attention_mask = torch.ones_like(input_ids)
        seq_lens = torch.tensor([7, 6, 5, 4])
        intent_labels = torch.randint(0, 60, (4,))
        slot_labels = torch.randint(0, 111, (4, 7))

        loss, intent_logits, slot_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            seq_lens=seq_lens,
            intent_labels=intent_labels,
            slot_labels=slot_labels,
        )

        self.assertIsNotNone(loss)
        self.assertEqual(intent_logits.shape, (4, 60))
        self.assertEqual(slot_logits.shape, (4, 7, 111))

    def test_dataloader_batch_contract(self):
        tmp_dir = Path("/tmp/stage1-baseline-tests")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        records = [
            {
                "partition": "train",
                "intent": "intent_a",
                "utt": "play hello",
                "annot_utt": "play [song : hello]",
            },
            {
                "partition": "dev",
                "intent": "intent_a",
                "utt": "play world",
                "annot_utt": "play [song : world]",
            },
            {
                "partition": "test",
                "intent": "intent_a",
                "utt": "play now",
                "annot_utt": "play [song : now]",
            },
        ]
        data_path = tmp_dir / "sample.jsonl"
        _write_jsonl(data_path, records)

        loaders, vocab, _ = get_dataloaders(
            data_path=str(data_path),
            batch_size=2,
            glove_path="",
            embedding_dim=16,
            enforce_official_counts=False,
        )

        batch = next(iter(loaders["train"]))
        self.assertEqual(
            set(batch.keys()),
            {
                "input_ids",
                "attention_mask",
                "seq_lens",
                "intent_label",
                "slot_labels",
                "raw_utt",
                "raw_annot_utt",
            },
        )
        self.assertEqual(batch["input_ids"].dim(), 2)
        self.assertEqual(batch["slot_labels"].shape, batch["input_ids"].shape)
        self.assertEqual(len(vocab.intent2id), 1)


if __name__ == "__main__":
    unittest.main()
