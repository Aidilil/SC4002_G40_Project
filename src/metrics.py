from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from .data_loader import tokenize_words


def _tags_from_ids(label_ids: Sequence[int], id2slot: Dict[int, str]) -> List[str]:
    return [id2slot[int(idx)] for idx in label_ids]


def bio_tags_to_spans(tokens: Sequence[str], tags: Sequence[str]) -> List[Tuple[str, str]]:
    spans: List[Tuple[str, str]] = []
    current_type = None
    current_tokens: List[str] = []

    def flush() -> None:
        nonlocal current_type, current_tokens
        if current_type and current_tokens:
            spans.append((current_type, " ".join(current_tokens)))
        current_type = None
        current_tokens = []

    for token, tag in zip(tokens, tags):
        if tag == "O" or tag == "<PAD>":
            flush()
            continue

        if "-" not in tag:
            flush()
            continue

        prefix, slot_type = tag.split("-", 1)
        if prefix == "B":
            flush()
            current_type = slot_type
            current_tokens = [token]
        elif prefix == "I" and current_type == slot_type:
            current_tokens.append(token)
        else:
            flush()
            current_type = slot_type
            current_tokens = [token]

    flush()
    return spans


def compute_slot_micro_f1(
    all_gold_spans: Iterable[List[Tuple[str, str]]],
    all_pred_spans: Iterable[List[Tuple[str, str]]],
) -> float:
    tp = fp = fn = 0

    for gold_spans, pred_spans in zip(all_gold_spans, all_pred_spans):
        gold_set = set(gold_spans)
        pred_set = set(pred_spans)

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    denominator = (2 * tp + fp + fn)
    if denominator == 0:
        return 0.0

    return 100.0 * (2 * tp) / denominator


def compute_intent_accuracy(gold: List[int], pred: List[int]) -> float:
    if not gold:
        return 0.0
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return 100.0 * correct / len(gold)


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    id2slot: Dict[int, str],
) -> Dict[str, float]:
    model.eval()

    all_gold_intents: List[int] = []
    all_pred_intents: List[int] = []
    all_gold_spans: List[List[Tuple[str, str]]] = []
    all_pred_spans: List[List[Tuple[str, str]]] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            seq_lens = batch["seq_lens"].to(device)

            _, intent_logits, slot_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                seq_lens=seq_lens,
            )

            pred_intents = intent_logits.argmax(dim=-1).cpu().tolist()
            gold_intents = batch["intent_label"].cpu().tolist()
            all_pred_intents.extend(pred_intents)
            all_gold_intents.extend(gold_intents)

            pred_slot_ids = slot_logits.argmax(dim=-1).cpu()
            gold_slot_ids = batch["slot_labels"].cpu()

            for i, seq_len in enumerate(batch["seq_lens"].tolist()):
                tokens = tokenize_words(batch["raw_utt"][i])
                tokens = tokens[:seq_len]

                pred_tags = _tags_from_ids(pred_slot_ids[i, :seq_len].tolist(), id2slot)
                gold_tags = _tags_from_ids(gold_slot_ids[i, :seq_len].tolist(), id2slot)

                all_pred_spans.append(bio_tags_to_spans(tokens, pred_tags))
                all_gold_spans.append(bio_tags_to_spans(tokens, gold_tags))

    intent_accuracy = compute_intent_accuracy(all_gold_intents, all_pred_intents)
    slot_micro_f1 = compute_slot_micro_f1(all_gold_spans, all_pred_spans)

    return {
        "intent_accuracy": round(intent_accuracy, 2),
        "slot_micro_f1": round(slot_micro_f1, 2),
        "combined_score": round((intent_accuracy + slot_micro_f1) / 2.0, 2),
    }
