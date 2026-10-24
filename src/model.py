from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointBiLSTMAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_intents: int,
        num_slots: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        slot_loss_weight: float = 1.0,
        embedding_matrix: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.slot_loss_weight = slot_loss_weight

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=freeze_embeddings, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        projection_dim = hidden_dim * 2
        self.attn_score = nn.Linear(projection_dim, 1)
        self.intent_dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(projection_dim, num_intents)
        self.slot_head = nn.Linear(projection_dim, num_slots)

        self._assert_parameter_budget()

    def _assert_parameter_budget(self) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        if total_params > 20_000_000:
            raise AssertionError(f"Parameter budget exceeded: {total_params} > 20,000,000")
        print(f"Total model parameters: {total_params}")

    def _attention_pool(self, encoded: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        scores = self.attn_score(encoded).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), encoded).squeeze(1)
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        intent_labels: Optional[torch.Tensor] = None,
        slot_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        embeddings = self.embedding(input_ids)

        if seq_lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embeddings, seq_lens.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.encoder(packed)
            encoded, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=input_ids.shape[1]
            )
        else:
            encoded, _ = self.encoder(embeddings)

        pooled = self._attention_pool(encoded, attention_mask)
        intent_logits = self.intent_head(self.intent_dropout(pooled))
        slot_logits = self.slot_head(encoded)

        loss = None
        if intent_labels is not None and slot_labels is not None:
            intent_loss = F.cross_entropy(intent_logits, intent_labels)
            slot_loss = F.cross_entropy(
                slot_logits.view(-1, slot_logits.shape[-1]),
                slot_labels.view(-1),
                ignore_index=0,
            )
            loss = intent_loss + self.slot_loss_weight * slot_loss

        return loss, intent_logits, slot_logits
