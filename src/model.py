import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _log_sum_exp(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Numerically stable log-sum-exp along a specified dimension.
    Prevents floating point overflow by subtracting the maximum value.
    """
    reachable = torch.isfinite(tensor).any(dim=dim, keepdim=True)
    safe = torch.where(reachable, tensor, torch.zeros_like(tensor))
    result = torch.logsumexp(safe, dim=dim)
    return result.masked_fill(~reachable.squeeze(dim), float('-inf'))


class LinearChainCRF(nn.Module):
    """
    Linear-chain Conditional Random Field (CRF) for slot sequence tagging.

    Learns transition preferences, with hard BIO constraints when id2slot
    is supplied. Omitting the mapping supports legacy unconstrained models.
    """

    def __init__(self, num_tags: int, id2slot: Optional[Dict[int, str]] = None) -> None:
        super().__init__()
        self.num_tags = num_tags
        allowed = torch.ones(num_tags, num_tags, dtype=torch.bool)
        start = torch.ones(num_tags, dtype=torch.bool)
        end = torch.ones(num_tags, dtype=torch.bool)
        if id2slot is not None:
            if set(id2slot) != set(range(num_tags)) or id2slot.get(0) != '<PAD>':
                raise ValueError('id2slot must cover every tag ID with <PAD> at ID 0.')
            labels = [id2slot[i] for i in range(num_tags)]
            if len(set(labels)) != num_tags or 'O' not in labels:
                raise ValueError('Slot labels must be unique and include O.')
            for tag in labels[1:]:
                if tag != 'O' and (not tag.startswith(('B-', 'I-')) or not tag[2:]):
                    raise ValueError(f'Invalid BIO label: {tag}')
                if tag.startswith('I-') and 'B-' + tag[2:] not in labels:
                    raise ValueError(f'Missing B tag for {tag}')
            start[0] = end[0] = False
            allowed[0, :] = False
            allowed[:, 0] = False
            for j, tag in enumerate(labels):
                if tag.startswith('I-'):
                    start[j] = False
                    for i, previous in enumerate(labels):
                        allowed[i, j] = previous in ('B-' + tag[2:], tag)
        # Derived from the vocabulary; do not change checkpoint tensor keys.
        self.register_buffer('allowed_transitions', allowed, persistent=False)
        self.register_buffer('allowed_start', start, persistent=False)
        self.register_buffer('allowed_end', end, persistent=False)

        # Transition score from tag i to tag j
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def _constrained_scores(self):
        return (self.transitions.masked_fill(~self.allowed_transitions, float('-inf')),
                self.start_transitions.masked_fill(~self.allowed_start, float('-inf')),
                self.end_transitions.masked_fill(~self.allowed_end, float('-inf')))

    def _validate(self, emissions, mask, tags=None):
        if emissions.ndim != 3 or emissions.shape[-1] != self.num_tags or emissions.shape[1] == 0:
            raise ValueError('Expected nonempty emissions [batch, length, num_tags].')
        if mask.shape != emissions.shape[:2] or not ((mask == 0) | (mask == 1)).all():
            raise ValueError('Mask must match emissions and contain only 0/1.')
        mask = mask.bool()
        if not mask[:, 0].all() or (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError('Each sequence must be nonempty and right-padded.')
        if tags is not None:
            if tags.shape != mask.shape or ((tags < 0) | (tags >= self.num_tags)).any():
                raise ValueError('Gold tags must match mask shape and be valid IDs.')
            if not self.allowed_start[tags[:, 0]].all():
                raise ValueError('Gold sequence has an invalid BIO start.')
            valid = self.allowed_transitions[tags[:, :-1], tags[:, 1:]]
            if (~valid & mask[:, 1:]).any():
                raise ValueError('Gold sequence has an invalid BIO transition or PAD tag.')
            last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
            if not self.allowed_end[last].all():
                raise ValueError('Gold sequence ends with an invalid tag.')
        return mask

    def reset_parameters(self) -> None:
        """Initialize transition parameters uniformly."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for a batch of sequences.

        Args:
            emissions: Emission scores of shape [batch_size, seq_len, num_tags].
            tags: Target tag indices of shape [batch_size, seq_len].
            mask: Binary tensor of shape [batch_size, seq_len] (1 for valid tokens, 0 for pad).

        Returns:
            Scalar tensor representing mean negative log-likelihood.
        """
        mask = self._validate(emissions, mask, tags)
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._compute_partition(emissions, mask)
        return (forward_score - gold_score).mean()

    def _score_sentence(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute the unnormalized path score for the gold tag sequence."""
        batch_size, seq_len, _ = emissions.shape
        transitions, start, end = self._constrained_scores()
        score = torch.zeros(batch_size, device=emissions.device)

        # Transition from start state to first token tag
        first_tags = tags[:, 0]
        score = score + start[first_tags]

        for t in range(seq_len):
            m_t = mask[:, t].bool()
            curr_tags = tags[:, t]

            # Token emission score
            emit_score = emissions[torch.arange(batch_size), t, curr_tags]
            score = score + torch.where(m_t, emit_score, 0.0)

            # Tag transition score
            if t > 0:
                prev_tags = tags[:, t - 1]
                trans_score = transitions[prev_tags, curr_tags]
                score = score + torch.where(m_t, trans_score, 0.0)

        # Transition from the last valid token to the end state
        seq_lens = mask.long().sum(dim=1)
        last_indices = torch.clamp(seq_lens - 1, min=0)
        last_tags = tags[torch.arange(batch_size), last_indices]
        score = score + end[last_tags]

        return score

    def _compute_partition(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute log-sum-exp over all valid paths using the forward algorithm."""
        batch_size, seq_len, _ = emissions.shape
        transitions, start, end = self._constrained_scores()
        alpha = start.unsqueeze(0) + emissions[:, 0]

        for t in range(1, seq_len):
            m_t = mask[:, t].unsqueeze(1).bool()
            # Broadcast alpha [B, K, 1] + transitions [1, K, K] -> [B, K, K]
            scores = alpha.unsqueeze(2) + transitions.unsqueeze(0)
            next_alpha = _log_sum_exp(scores, dim=1) + emissions[:, t]
            # Only update state if position is not padding
            alpha = torch.where(m_t, next_alpha, alpha)

        alpha = alpha + end.unsqueeze(0)
        return _log_sum_exp(alpha, dim=1)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Find the highest-scoring tag sequence using Viterbi decoding.

        Args:
            emissions: Emission scores of shape [batch_size, seq_len, num_tags].
            mask: Binary tensor of shape [batch_size, seq_len].

        Returns:
            Decoded tag indices of shape [batch_size, seq_len].
        """
        mask = self._validate(emissions, mask)
        batch_size, seq_len, _ = emissions.shape
        transitions, start, end = self._constrained_scores()
        viterbi = start.unsqueeze(0) + emissions[:, 0]
        backpointers = []

        for t in range(1, seq_len):
            m_t = mask[:, t].unsqueeze(1).bool()
            scores = viterbi.unsqueeze(2) + transitions.unsqueeze(0)
            max_scores, best_tags = scores.max(dim=1)
            next_viterbi = max_scores + emissions[:, t]
            viterbi = torch.where(m_t, next_viterbi, viterbi)
            backpointers.append(best_tags)

        viterbi = viterbi + end.unsqueeze(0)
        best_last_tags = viterbi.argmax(dim=1).tolist()
        seq_lens = mask.long().sum(dim=1).tolist()

        best_paths = []
        for b in range(batch_size):
            length = seq_lens[b]
            if length <= 0:
                best_paths.append([0] * seq_len)
                continue

            best_tag = best_last_tags[b]
            path = [best_tag]
            for t in range(length - 2, -1, -1):
                best_tag = backpointers[t][b, best_tag].item()
                path.append(best_tag)
            path.reverse()
            path = path + [0] * (seq_len - length)
            best_paths.append(path)

        return torch.tensor(best_paths, dtype=torch.long, device=emissions.device)


class JointBiLSTMAttention(nn.Module):
    """
    Joint BiLSTM model for simultaneous intent classification and slot filling.

    The architecture shares a bidirectional LSTM encoder across two heads:
      1. Intent head: pools contextual token embeddings using learned linear
         attention scores into an utterance vector, then classifies the intent.
      2. Slot head: projects hidden states to tag emissions and uses a linear-chain
         CRF to decode the highest-scoring tag sequence.
    """

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
        use_crf: bool = True,
        id2slot: Optional[Dict[int, str]] = None,
        enforce_bio: bool = True,
    ) -> None:
        super().__init__()
        if not math.isfinite(slot_loss_weight) or slot_loss_weight < 0:
            raise ValueError('slot_loss_weight must be finite and nonnegative.')
        self.slot_loss_weight = slot_loss_weight
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_crf = use_crf

        # Word embedding layer
        if embedding_matrix is not None:
            if tuple(embedding_matrix.shape) != (vocab_size, embedding_dim):
                raise ValueError('Embedding matrix must have shape [vocab_size, embedding_dim].')
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=freeze_embeddings, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.embedding.weight.requires_grad_(not freeze_embeddings)

        # Shared bidirectional LSTM encoder
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        projection_dim = hidden_dim * 2

        # Intent classification branch
        self.attn_score = nn.Linear(projection_dim, 1)
        self.intent_dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(projection_dim, num_intents)

        # Slot filling branch
        self.slot_dropout = nn.Dropout(dropout)
        self.slot_head = nn.Linear(projection_dim, num_slots)

        # Linear-chain CRF for slot transitions
        if use_crf and enforce_bio and id2slot is None:
            raise ValueError('Constrained CRF requires id2slot; use enforce_bio=False only for legacy runs.')
        self.crf = LinearChainCRF(num_slots, id2slot if enforce_bio else None) if use_crf else None

        self._check_parameter_budget()

    def get_parameter_counts(self) -> Dict[str, int]:
        """Return total, trainable, and frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}

    def _check_parameter_budget(self) -> None:
        """Verify model parameter count stays within course limits (<= 20M)."""
        counts = self.get_parameter_counts()
        total_params = counts["total"]
        if total_params > 20_000_000:
            raise AssertionError(
                f"Model exceeds 20M parameter ceiling: {total_params:,} parameters."
            )

    def _attention_pool(
        self, encoded: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Softmax-normalize learned token scores and pool token states."""
        scores = self.attn_score(encoded).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=1)
        return torch.bmm(attn_weights.unsqueeze(1), encoded).squeeze(1)

    @torch.no_grad()
    def decode_slots(
        self, slot_logits: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return tag IDs [batch, length] from emissions, with padding ID 0.

        Uses Viterbi when a CRF is enabled, otherwise token-wise argmax.
        Call eval() before forward() for deterministic inference without dropout.
        """
        if self.crf is not None:
            return self.crf.decode(slot_logits, attention_mask)
        return slot_logits.argmax(dim=-1).masked_fill(attention_mask == 0, 0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        intent_labels: Optional[torch.Tensor] = None,
        slot_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Run forward pass for intent classification and slot labeling.

        Args:
            input_ids: Token IDs of shape [batch_size, max_seq_len].
            attention_mask: Mask of shape [batch_size, max_seq_len] (1 for tokens, 0 for pad).
            seq_lens: Sequence lengths before padding of shape [batch_size].
            intent_labels: Ground-truth intent labels of shape [batch_size].
            slot_labels: Ground-truth slot labels of shape [batch_size, max_seq_len].

        Returns:
            Tuple of (loss, intent_logits, slot_logits):
              - loss: Multi-task loss scalar if any labels are provided, else None.
              - intent_logits: Shape [batch_size, num_intents].
              - slot_logits: Genuine emission scores, shape
                [batch_size, max_seq_len, num_slots], with or without labels.
                Use decode_slots(slot_logits, attention_mask) for predicted tags.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError('input_ids must be a nonempty [batch, length] tensor.')
        token_mask = input_ids != 0
        if attention_mask is None:
            attention_mask = token_mask
        elif (attention_mask.shape != input_ids.shape
              or not ((attention_mask == 0) | (attention_mask == 1)).all()
              or not torch.equal(attention_mask.bool(), token_mask)):
            raise ValueError('attention_mask must mark precisely the non-PAD input tokens.')
        attention_mask = attention_mask.bool()
        if not attention_mask[:, 0].all() or (attention_mask[:, 1:] & ~attention_mask[:, :-1]).any():
            raise ValueError('Inputs must contain nonempty, right-padded sequences.')
        inferred_lengths = attention_mask.long().sum(1)
        if seq_lens is not None and not torch.equal(seq_lens.to(inferred_lengths.device), inferred_lengths):
            raise ValueError('seq_lens must equal the number of real tokens in each sequence.')
        seq_lens = inferred_lengths
        embeddings = self.embedding(input_ids)

        # Efficient sequence packing to ignore padding steps
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

        # 1. Intent predictions
        pooled = self._attention_pool(encoded, attention_mask)
        intent_logits = self.intent_head(self.intent_dropout(pooled))

        # 2. Slot emission logits
        slot_logits = self.slot_head(self.slot_dropout(encoded))

        # 3. Multi-task loss
        loss = None
        if intent_labels is not None or slot_labels is not None:
            total_loss = torch.tensor(0.0, device=input_ids.device)

            if intent_labels is not None:
                total_loss = total_loss + F.cross_entropy(intent_logits, intent_labels)

            if slot_labels is not None:
                if self.crf is not None:
                    mask = (
                        attention_mask
                        if attention_mask is not None
                        else (input_ids != 0).long()
                    )
                    slot_loss = self.crf(slot_logits, slot_labels, mask)
                else:
                    slot_loss = F.cross_entropy(
                        slot_logits.view(-1, slot_logits.shape[-1]),
                        slot_labels.view(-1),
                        ignore_index=0,
                    )
                total_loss = total_loss + self.slot_loss_weight * slot_loss

            loss = total_loss

        return loss, intent_logits, slot_logits
