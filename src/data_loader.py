import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import assert_official_split_counts

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+|[^\w\s]")
_SLOT_PATTERN = re.compile(r"\[\s*([^\[\]:]+?)\s*:\s*([^\]]+?)\s*\]")


@dataclass
class Vocabulary:
    word2id: Dict[str, int]
    intent2id: Dict[str, int]
    slot2id: Dict[str, int]


class MassiveDataset(Dataset):
    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


def tokenize_words(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text)


def parse_annot_utt(annot_utt: str) -> Tuple[List[str], List[str]]:
    tokens: List[str] = []
    tags: List[str] = []
    cursor = 0

    for match in _SLOT_PATTERN.finditer(annot_utt):
        prefix = annot_utt[cursor:match.start()]
        for token in tokenize_words(prefix):
            tokens.append(token)
            tags.append("O")

        slot_type = match.group(1).strip()
        slot_value = match.group(2).strip()
        slot_tokens = tokenize_words(slot_value)

        for i, token in enumerate(slot_tokens):
            tokens.append(token)
            tags.append(f"B-{slot_type}" if i == 0 else f"I-{slot_type}")

        cursor = match.end()

    suffix = annot_utt[cursor:]
    for token in tokenize_words(suffix):
        tokens.append(token)
        tags.append("O")

    return tokens, tags


def load_massive_jsonl(data_path: str) -> List[Dict]:
    records: List[Dict] = []
    with Path(data_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_label_vocabs(train_records: List[Dict], enforce_official_counts: bool = True) -> Tuple[Dict[str, int], Dict[str, int]]:
    intent_labels = sorted({str(record["intent"]) for record in train_records})

    slot_types = set()
    for record in train_records:
        _, tags = parse_annot_utt(str(record["annot_utt"]))
        for tag in tags:
            if tag.startswith("B-"):
                slot_types.add(tag[2:])
            elif tag.startswith("I-"):
                slot_types.add(tag[2:])

    sorted_slot_types = sorted(slot_types)

    intent2id = {intent: idx for idx, intent in enumerate(intent_labels)}

    slot2id = {"<PAD>": 0, "O": 1}
    index = 2
    for slot_type in sorted_slot_types:
        slot2id[f"B-{slot_type}"] = index
        index += 1
        slot2id[f"I-{slot_type}"] = index
        index += 1

    if enforce_official_counts:
        if len(intent2id) != 60:
            raise AssertionError(f"Expected 60 intents, got {len(intent2id)}")
        if len(slot2id) != 111:
            raise AssertionError(f"Expected 111 slot tags, got {len(slot2id)}")

    return intent2id, slot2id


def build_word_vocab(train_records: List[Dict]) -> Dict[str, int]:
    vocab_words = set()
    for record in train_records:
        for token in tokenize_words(str(record["utt"])):
            vocab_words.add(token.lower())

    sorted_words = sorted(vocab_words)
    word2id = {"<PAD>": 0, "<UNK>": 1}
    for idx, word in enumerate(sorted_words, start=2):
        word2id[word] = idx

    return word2id


def load_glove_embeddings(glove_path: str, word2id: Dict[str, int], embedding_dim: int) -> torch.Tensor:
    embeddings = np.random.normal(0.0, 0.02, size=(len(word2id), embedding_dim)).astype(np.float32)
    embeddings[word2id["<PAD>"]] = np.zeros(embedding_dim, dtype=np.float32)

    if glove_path:
        glove_vectors: Dict[str, np.ndarray] = {}
        with Path(glove_path).open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != embedding_dim + 1:
                    continue
                word = parts[0]
                vector = np.asarray(parts[1:], dtype=np.float32)
                glove_vectors[word] = vector

        if glove_vectors:
            all_vectors = np.stack(list(glove_vectors.values()), axis=0)
            mean_vector = all_vectors.mean(axis=0)
        else:
            mean_vector = np.zeros(embedding_dim, dtype=np.float32)

        embeddings[word2id["<UNK>"]] = mean_vector
        for word, idx in word2id.items():
            if word in glove_vectors:
                embeddings[idx] = glove_vectors[word]

    return torch.tensor(embeddings, dtype=torch.float32)


def _encode_example(record: Dict, vocab: Vocabulary) -> Dict:
    parsed_tokens, slot_tags = parse_annot_utt(str(record["annot_utt"]))
    tokens = parsed_tokens if parsed_tokens else tokenize_words(str(record["utt"]))

    input_ids = [vocab.word2id.get(token.lower(), vocab.word2id["<UNK>"]) for token in tokens]
    slot_labels = [vocab.slot2id.get(tag, vocab.slot2id["O"]) for tag in slot_tags]

    if len(slot_labels) != len(input_ids):
        min_len = min(len(slot_labels), len(input_ids))
        input_ids = input_ids[:min_len]
        slot_labels = slot_labels[:min_len]

    return {
        "input_ids": input_ids,
        "intent_label": vocab.intent2id[str(record["intent"])],
        "slot_labels": slot_labels,
        "raw_utt": str(record["utt"]),
        "raw_annot_utt": str(record["annot_utt"]),
    }


def _collate_batch(batch: List[Dict]) -> Dict:
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    slot_labels = []
    attention_mask = []
    seq_lens = []
    intent_labels = []
    raw_utt = []
    raw_annot_utt = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        input_ids.append(item["input_ids"] + [0] * pad_len)
        slot_labels.append(item["slot_labels"] + [0] * pad_len)
        attention_mask.append([1] * seq_len + [0] * pad_len)
        seq_lens.append(seq_len)
        intent_labels.append(item["intent_label"])
        raw_utt.append(item["raw_utt"])
        raw_annot_utt.append(item["raw_annot_utt"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "seq_lens": torch.tensor(seq_lens, dtype=torch.long),
        "intent_label": torch.tensor(intent_labels, dtype=torch.long),
        "slot_labels": torch.tensor(slot_labels, dtype=torch.long),
        "raw_utt": raw_utt,
        "raw_annot_utt": raw_annot_utt,
    }


def get_dataloaders(
    data_path: str,
    batch_size: int,
    glove_path: str = "",
    embedding_dim: int = 300,
    enforce_official_counts: bool = True,
) -> Tuple[Dict[str, DataLoader], Vocabulary, torch.Tensor]:
    records = load_massive_jsonl(data_path)
    if enforce_official_counts:
        assert_official_split_counts(records)

    split_records: Dict[str, List[Dict]] = {"train": [], "dev": [], "test": []}
    for record in records:
        partition = str(record.get("partition", "")).lower()
        if partition in split_records:
            split_records[partition].append(record)

    intent2id, slot2id = build_label_vocabs(
        split_records["train"], enforce_official_counts=enforce_official_counts
    )
    word2id = build_word_vocab(split_records["train"])
    vocab = Vocabulary(word2id=word2id, intent2id=intent2id, slot2id=slot2id)

    loaders: Dict[str, DataLoader] = {}
    for split_name, split_data in split_records.items():
        encoded = [_encode_example(record, vocab) for record in split_data]
        dataset = MassiveDataset(encoded)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            collate_fn=_collate_batch,
        )

    embedding_matrix = load_glove_embeddings(glove_path, word2id, embedding_dim)
    return loaders, vocab, embedding_matrix
