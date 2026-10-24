import argparse
import json
from pathlib import Path

import torch
from torch.optim import Adam, AdamW
from tqdm import tqdm

from src.data_loader import get_dataloaders
from src.metrics import evaluate_model
from src.model import JointBiLSTMAttention
from src.utils import ExperimentTracker, set_seed


def load_config(config_path: str):
    with Path(config_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def make_optimizer(model, config):
    optimizer_name = config["training"]["optimizer"].lower()
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"].get("weight_decay", 0.0)

    if optimizer_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            loss, _, _ = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                seq_lens=batch["seq_lens"].to(device),
                intent_labels=batch["intent_label"].to(device),
                slot_labels=batch["slot_labels"].to(device),
            )
            total_loss += float(loss.item())
            steps += 1

    return total_loss / max(1, steps)


def train(config_path: str):
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))

    trackers = ExperimentTracker(config["experiment_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders, vocab, embedding_matrix = get_dataloaders(
        data_path=config["data_path"],
        batch_size=int(config["training"]["batch_size"]),
        glove_path=config["embedding"].get("glove_path", ""),
        embedding_dim=int(config["embedding"]["embedding_dim"]),
        enforce_official_counts=True,
    )

    model = JointBiLSTMAttention(
        vocab_size=len(vocab.word2id),
        embedding_dim=int(config["embedding"]["embedding_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_intents=len(vocab.intent2id),
        num_slots=len(vocab.slot2id),
        num_layers=int(config["model"].get("num_layers", 2)),
        dropout=float(config["training"].get("dropout", 0.3)),
        slot_loss_weight=float(config["training"].get("slot_loss_weight", 1.0)),
        embedding_matrix=embedding_matrix,
        freeze_embeddings=bool(config["embedding"].get("freeze", False)),
    ).to(device)

    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    id2slot = {idx: slot for slot, idx in vocab.slot2id.items()}

    best_score = float("-inf")
    best_epoch = -1
    best_metrics = None

    for epoch in range(1, int(config["training"]["num_epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(dataloaders["train"], desc=f"Epoch {epoch}", leave=False)

        for batch in progress:
            optimizer.zero_grad()

            loss, _, _ = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                seq_lens=batch["seq_lens"].to(device),
                intent_labels=batch["intent_label"].to(device),
                slot_labels=batch["slot_labels"].to(device),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(config["training"].get("max_grad_norm", 1.0))
            )
            optimizer.step()

            epoch_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / max(1, len(dataloaders["train"]))
        dev_loss = evaluate_loss(model, dataloaders["dev"], device)
        dev_metrics = evaluate_model(model, dataloaders["dev"], device, id2slot)
        scheduler.step(dev_metrics["combined_score"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            **dev_metrics,
        }
        trackers.append_metrics(row)
        trackers.log(
            "epoch={epoch} train_loss={train:.4f} dev_loss={dev:.4f} "
            "intent_acc={intent:.2f}% slot_f1={slot:.2f}% combined={combined:.2f}".format(
                epoch=epoch,
                train=train_loss,
                dev=dev_loss,
                intent=dev_metrics["intent_accuracy"],
                slot=dev_metrics["slot_micro_f1"],
                combined=dev_metrics["combined_score"],
            )
        )

        if dev_metrics["combined_score"] > best_score:
            best_score = dev_metrics["combined_score"]
            best_epoch = epoch
            best_metrics = dev_metrics
            trackers.save_checkpoint(model, optimizer, epoch, best_score)

    checkpoint = torch.load(trackers.best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    reloaded_dev_metrics = evaluate_model(model, dataloaders["dev"], device, id2slot)

    if reloaded_dev_metrics["combined_score"] != round(best_score, 2):
        raise AssertionError(
            "Reloaded dev score mismatch: "
            f"{reloaded_dev_metrics['combined_score']} != {round(best_score, 2)}"
        )

    test_metrics = evaluate_model(model, dataloaders["test"], device, id2slot)
    trackers.log(
        "best_epoch={epoch} best_dev={best:.2f} checkpoint={ckpt}".format(
            epoch=best_epoch,
            best=best_score,
            ckpt=trackers.best_checkpoint_path.name,
        )
    )
    trackers.log(
        "test_intent_acc={intent:.2f}% test_slot_f1={slot:.2f}% test_combined={combined:.2f}".format(
            intent=test_metrics["intent_accuracy"],
            slot=test_metrics["slot_micro_f1"],
            combined=test_metrics["combined_score"],
        )
    )

    return {
        "best_epoch": best_epoch,
        "best_dev_metrics": best_metrics,
        "reloaded_dev_metrics": reloaded_dev_metrics,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SC4002 baseline model")
    parser.add_argument("--config", default="params.json", type=str)
    args = parser.parse_args()
    train(args.config)
