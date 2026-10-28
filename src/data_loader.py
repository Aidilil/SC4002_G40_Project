import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import assert_official_split_counts


# Unicode-friendly tokenization
_TOKEN_PATTERN = re.compile(r"\w+(?:'\w+)?|[^\w\s]")

_SLOT_PATTERN = re.compile(
    r"\[\s*([^\[\]:]+?)\s*:\s*([^\]]+?)\s*\]"
)


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
            tags.append(
                f"B-{slot_type}" if i == 0 else f"I-{slot_type}"
            )

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


def build_label_vocabs(
    train_records: List[Dict],
    enforce_official_counts: bool = True,
) -> Tuple[Dict[str, int], Dict[str, int]]:

    intent_labels = sorted(
        {str(record["intent"]) for record in train_records}
    )

    slot_types = set()

    for record in train_records:
        _, tags = parse_annot_utt(str(record["annot_utt"]))

        for tag in tags:
            if tag.startswith("B-") or tag.startswith("I-"):
                slot_types.add(tag[2:])

    sorted_slot_types = sorted(slot_types)

    intent2id = {
        intent: idx
        for idx, intent in enumerate(intent_labels)
    }

    slot2id = {
        "<PAD>": 0,
        "O": 1,
    }

    index = 2

    for slot_type in sorted_slot_types:
        slot2id[f"B-{slot_type}"] = index
        index += 1

        slot2id[f"I-{slot_type}"] = index
        index += 1

    if enforce_official_counts:

        if len(intent2id) != 60:
            raise AssertionError(
                f"Expected 60 intents, got {len(intent2id)}"
            )

        if len(slot_types) != 55:
            raise AssertionError(
                f"Expected 55 slot types, got {len(slot_types)}"
            )

    return intent2id, slot2id


def build_word_vocab(train_records: List[Dict]) -> Dict[str, int]:

    vocab_words = set()

    for record in train_records:
        for token in tokenize_words(str(record["utt"])):
            vocab_words.add(token.lower())

    sorted_words = sorted(vocab_words)

    word2id = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for idx, word in enumerate(sorted_words, start=2):
        word2id[word] = idx

    return word2id


def load_glove_embeddings(
    glove_path: str,
    word2id: Dict[str, int],
    embedding_dim: int,
) -> torch.Tensor:

    embeddings = np.random.normal(
        0.0,
        0.02,
        size=(len(word2id), embedding_dim),
    ).astype(np.float32)

    embeddings[word2id["<PAD>"]] = np.zeros(
        embedding_dim,
        dtype=np.float32,
    )

    if glove_path:

        all_vectors = []

        with Path(glove_path).open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:
                parts = line.strip().split()

                if len(parts) != embedding_dim + 1:
                    continue

                word = parts[0]

                vector = np.asarray(
                    parts[1:],
                    dtype=np.float32,
                )

                all_vectors.append(vector)

                if word in word2id:
                    embeddings[word2id[word]] = vector

        if all_vectors:
            mean_vector = np.mean(
                all_vectors,
                axis=0,
            )
        else:
            mean_vector = np.zeros(
                embedding_dim,
                dtype=np.float32,
            )

        embeddings[word2id["<UNK>"]] = mean_vector

    return torch.tensor(
        embeddings,
        dtype=torch.float32,
    )


def _encode_example(
    record: Dict,
    vocab: Vocabulary,
) -> Dict:

    parsed_tokens, slot_tags = parse_annot_utt(
        str(record["annot_utt"])
    )

    tokens = (
        parsed_tokens
        if parsed_tokens
        else tokenize_words(str(record["utt"]))
    )

    input_ids = [
        vocab.word2id.get(
            token.lower(),
            vocab.word2id["<UNK>"],
        )
        for token in tokens
    ]

    slot_labels = [
        vocab.slot2id.get(
            tag,
            vocab.slot2id["O"],
        )
        for tag in slot_tags
    ]

    if len(slot_labels) != len(input_ids):
        raise ValueError(
            f"Token-label mismatch\n"
            f"utt={record['utt']}\n"
            f"annot_utt={record['annot_utt']}\n"
            f"tokens={len(input_ids)}\n"
            f"labels={len(slot_labels)}"
        )

    return {
        "input_ids": input_ids,
        "intent_label": vocab.intent2id[
            str(record["intent"])
        ],
        "slot_labels": slot_labels,
        "raw_utt": str(record["utt"]),
        "raw_annot_utt": str(record["annot_utt"]),
    }


def _collate_batch(batch: List[Dict]) -> Dict:

    input_ids = [
        torch.tensor(
            item["input_ids"],
            dtype=torch.long,
        )
        for item in batch
    ]

    slot_labels = [
        torch.tensor(
            item["slot_labels"],
            dtype=torch.long,
        )
        for item in batch
    ]

    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0,
    )

    padded_slot_labels = torch.nn.utils.rnn.pad_sequence(
        slot_labels,
        batch_first=True,
        padding_value=0,
    )

    attention_mask = (
        padded_input_ids != 0
    ).long()

    seq_lens = torch.tensor(
        [len(item["input_ids"]) for item in batch],
        dtype=torch.long,
    )

    intent_labels = torch.tensor(
        [item["intent_label"] for item in batch],
        dtype=torch.long,
    )

    raw_utt = [
        item["raw_utt"]
        for item in batch
    ]

    raw_annot_utt = [
        item["raw_annot_utt"]
        for item in batch
    ]

    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "seq_lens": seq_lens,
        "intent_label": intent_labels,
        "slot_labels": padded_slot_labels,
        "raw_utt": raw_utt,
        "raw_annot_utt": raw_annot_utt,
    }


def get_dataloaders(
    data_path: str,
    batch_size: int,
    glove_path: str = "",
    embedding_dim: int = 300,
    enforce_official_counts: bool = True,
):

    records = load_massive_jsonl(data_path)

    if enforce_official_counts:
        assert_official_split_counts(records)

    split_records = {
        "train": [],
        "dev": [],
        "test": [],
    }

    for record in records:
        partition = str(record.get("partition", "")).lower()

        if partition in split_records:
            split_records[partition].append(record)

    intent2id, slot2id = build_label_vocabs(
        split_records["train"],
        enforce_official_counts=
        enforce_official_counts,
    )

    word2id = build_word_vocab(
        split_records["train"]
    )

    vocab = Vocabulary(
        word2id=word2id,
        intent2id=intent2id,
        slot2id=slot2id,
    )

    loaders = {}

    for split_name, split_data in split_records.items():

        encoded = [
            _encode_example(record, vocab)
            for record in split_data
        ]

        dataset = MassiveDataset(encoded)

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            collate_fn=_collate_batch,
        )

    embedding_matrix = load_glove_embeddings(
        glove_path,
        word2id,
        embedding_dim,
    )

    return loaders, vocab, embedding_matrix