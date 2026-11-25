"""Tensor-only WAM prototype. Browser identities and raw text stay outside it."""

import dataclasses
import math

import torch
from torch import nn
from torch.nn import functional


SLOT_DIM = 256
MECHANISM_DIM = 64
STATE_DIM = 64
CONTENT_DIM = 128
TEXT_DIM = 384
CONTEXT_DIM = 64
ACTION_DIM = 128
STATE_COUNT = 8
FLAG_COUNT = 4
ACTION_COUNT = 5
SLOT_COUNT = 128
HEAD_COUNT = 4


@dataclasses.dataclass(frozen=True)
class Belief:
    """Aligned slots. All floating fields can be fed back into imagination.

    latent: [B,N,256]; present: [B,N]; states: [B,N,8];
    roles: [B,N,R]; flags: [B,N,4]; boxes: [B,N,4];
    affordances: [B,N,4] (click/type/select/check); parents: [B,N,N+1]
    (index 0 = no parent); context: [B,64]. Probabilities, not hard labels.
    """

    latent: torch.Tensor
    present: torch.Tensor
    states: torch.Tensor
    roles: torch.Tensor
    flags: torch.Tensor
    boxes: torch.Tensor
    affordances: torch.Tensor
    parents: torch.Tensor
    context: torch.Tensor


@dataclasses.dataclass(frozen=True)
class Prediction:
    """Next belief and direction-specific logits for supervised training."""

    belief: Belief
    present_logits: torch.Tensor
    one_to_zero_logits: torch.Tensor
    zero_to_one_logits: torch.Tensor
    birth_state_logits: torch.Tensor
    role_logits: torch.Tensor
    flag_logits: torch.Tensor
    affordance_logits: torch.Tensor
    parent_logits: torch.Tensor
    content_source_logits: torch.Tensor


class RelationBlock(nn.Module):
    """Dense attention with parent and entity-presence bias."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(SLOT_DIM)
        self.attention = nn.MultiheadAttention(
            SLOT_DIM, HEAD_COUNT, batch_first=True, dropout=0.0
        )
        self.norm2 = nn.LayerNorm(SLOT_DIM)
        self.feedforward = nn.Sequential(
            nn.Linear(SLOT_DIM, 512), nn.SiLU(), nn.Linear(512, SLOT_DIM)
        )
        self.parent_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, latent: torch.Tensor, present: torch.Tensor,
                parents: torch.Tensor) -> torch.Tensor:
        parent_edges = parents[..., 1:]
        bias = self.parent_weight * (
            parent_edges + parent_edges.transpose(-1, -2)
        )
        bias = bias + present.clamp_min(1e-4).log().unsqueeze(1)
        batch_size, slots, _ = latent.shape
        mask = bias.unsqueeze(1).expand(-1, HEAD_COUNT, -1, -1)
        mask = mask.reshape(batch_size * HEAD_COUNT, slots, slots)
        normalized = self.norm1(latent)
        messages, _ = self.attention(
            normalized, normalized, normalized, attn_mask=mask,
            need_weights=False,
        )
        latent = latent + messages
        return latent + self.feedforward(self.norm2(latent))


class WorldActionModel(nn.Module):
    """Shared learned dynamics, evidence adaptation, and goal evaluation."""

    def __init__(self, role_count: int, slot_count: int = SLOT_COUNT) -> None:
        super().__init__()
        if role_count < 2 or slot_count < 2:
            raise ValueError('At least two roles and slots are required.')
        self.role_count = role_count
        self.slot_count = slot_count
        self.content = nn.Sequential(
            nn.Linear(2 * TEXT_DIM, CONTENT_DIM), nn.LayerNorm(CONTENT_DIM),
            nn.SiLU(),
        )
        self.mechanism = nn.Sequential(
            nn.Linear(role_count + CONTENT_DIM + FLAG_COUNT + 4, 64),
            nn.LayerNorm(64), nn.SiLU(),
        )
        self.state = nn.Sequential(
            nn.Linear(role_count + STATE_COUNT + FLAG_COUNT + 4 + 4, 64),
            nn.LayerNorm(64), nn.SiLU(),
        )
        self.empty_slots = nn.Parameter(torch.randn(slot_count, SLOT_DIM) * .02)
        self.observation_blocks = nn.ModuleList(RelationBlock() for _ in range(2))
        self.history_gate = nn.Linear(2 * SLOT_DIM, MECHANISM_DIM + STATE_DIM)
        self.action_kind = nn.Embedding(ACTION_COUNT, 32)
        self.action_encoder = nn.Sequential(
            nn.Linear(32 + SLOT_DIM + TEXT_DIM + 3 + 1 + 2, ACTION_DIM),
            nn.LayerNorm(ACTION_DIM), nn.SiLU(),
        )
        self.condition = nn.Linear(SLOT_DIM + CONTEXT_DIM + ACTION_DIM + 1,
                                   SLOT_DIM)
        self.structured_feedback = nn.Linear(
            1 + role_count + STATE_COUNT + FLAG_COUNT + 4 + 4,
            SLOT_DIM,
        )
        self.action_gate = nn.Linear(ACTION_DIM, SLOT_DIM)
        self.gated_projection = nn.Linear(SLOT_DIM, SLOT_DIM)
        self.dynamics_blocks = nn.ModuleList(RelationBlock() for _ in range(2))
        self.delta = nn.Linear(SLOT_DIM, SLOT_DIM)
        self.next_norm = nn.LayerNorm(SLOT_DIM)
        self.present_head = nn.Linear(SLOT_DIM, 1)
        self.flip_head = nn.Linear(SLOT_DIM, 2 * STATE_COUNT)
        self.birth_state_head = nn.Linear(SLOT_DIM, STATE_COUNT)
        self.role_head = nn.Linear(SLOT_DIM, role_count)
        self.flag_head = nn.Linear(SLOT_DIM, FLAG_COUNT)
        self.box_head = nn.Linear(SLOT_DIM, 4)
        self.affordance_head = nn.Linear(SLOT_DIM, 4)
        self.parent_query = nn.Linear(SLOT_DIM, 64)
        self.parent_key = nn.Linear(SLOT_DIM, 64)
        self.root_head = nn.Linear(SLOT_DIM, 1)
        self.evidence = nn.Sequential(
            nn.Linear(2 * SLOT_DIM + ACTION_DIM, CONTEXT_DIM), nn.SiLU()
        )
        self.adaptation = nn.GRUCell(CONTEXT_DIM, CONTEXT_DIM)
        self.goal_tokens = nn.Linear(TEXT_DIM, SLOT_DIM)
        self.goal_attention = nn.MultiheadAttention(
            SLOT_DIM, HEAD_COUNT, batch_first=True, dropout=0.0
        )
        self.goal_fusion = nn.Sequential(
            nn.Linear(2 * SLOT_DIM + STATE_COUNT + 1 + CONTEXT_DIM, SLOT_DIM),
            nn.LayerNorm(SLOT_DIM), nn.SiLU(),
        )
        self.goal_pool = nn.Linear(SLOT_DIM, 1)
        self.goal_heads = nn.ModuleList(nn.Linear(SLOT_DIM, 2) for _ in range(3))
        self.proposal_head = nn.Linear(SLOT_DIM, 4)
        self.copy_query = nn.Linear(SLOT_DIM, 2 * SLOT_DIM)
        self.inverse_features = nn.Sequential(
            nn.Linear(2 * SLOT_DIM, 128), nn.SiLU()
        )
        self.inverse_kind = nn.Linear(128, ACTION_COUNT)
        self.inverse_target = nn.Linear(128, 1)
        self.inverse_targetless = nn.Linear(128, 1)
        self.content_query = nn.Linear(SLOT_DIM, 128)
        self.content_key = nn.Linear(SLOT_DIM, 128)
        self.content_fixed = nn.Linear(SLOT_DIM, 10)

    def observe(self, *, names: torch.Tensor, values: torch.Tensor,
                roles: torch.Tensor, states: torch.Tensor, flags: torch.Tensor,
                boxes: torch.Tensor, affordances: torch.Tensor,
                parents: torch.Tensor, present: torch.Tensor,
                previous: Belief | None = None,
                matched: torch.Tensor | None = None) -> Belief:
        """Encodes aligned actual slots; all floats use model dtype.

        names/values [B,N,384], roles [B,N,R] one hot (observed), states
        [B,N,8], flags/boxes/affordances [B,N,4], parents [B,N,N+1],
        present [B,N]. Previous and matched must be supplied together;
        matcher is external and must use only already observed data.
        """
        if present.shape[1] != self.slot_count:
            raise ValueError('Observation slot count differs from model.')
        if (previous is None) != (matched is None):
            raise ValueError('Previous belief and matched mask are paired.')
        content = self.content(torch.cat((names, values), dim=-1))
        mechanism = self.mechanism(torch.cat((roles, content, flags, boxes), -1))
        state = self.state(torch.cat((roles, states, flags, boxes, affordances), -1))
        latent = torch.cat((mechanism, state, content), dim=-1)
        latent = torch.where(present.bool().unsqueeze(-1), latent,
                             self.empty_slots.unsqueeze(0))
        if previous is not None:
            history = previous.latent[..., :128]
            gate = self.history_gate(torch.cat((latent, previous.latent), -1))
            gate = gate.sigmoid() * matched.unsqueeze(-1).to(latent.dtype)
            current = latent[..., :128]
            latent = torch.cat((current + gate * (history - current),
                                latent[..., 128:]), -1)
        for block in self.observation_blocks:
            latent = block(latent, present, parents)
        # Do not let contextual attention overwrite the direct content slice.
        latent = torch.cat((latent[..., :128], torch.where(
            present.bool().unsqueeze(-1), content, self.empty_slots[:, 128:]
        )), -1)
        context = (latent.new_zeros((latent.shape[0], CONTEXT_DIM))
                   if previous is None else previous.context)
        return Belief(latent, present, states, roles, flags, boxes, affordances,
                      parents, context)

    def encode_action(self, belief: Belief, kind: torch.Tensor,
                      target: torch.Tensor, text: torch.Tensor,
                      checked: torch.Tensor, scroll: torch.Tensor,
                      pointer: torch.Tensor) -> torch.Tensor:
        """kind/target [B] long; -1 is targetless; text [B,384];
        checked [B,3] one hot (unset/false/true); scroll [B,1]; pointer [B,2].
        """
        batch_index = torch.arange(target.shape[0], device=target.device)
        target_features = belief.latent[batch_index, target.clamp_min(0)]
        target_features = target_features * (target >= 0).unsqueeze(-1)
        features = torch.cat((self.action_kind(kind), target_features, text,
                              checked, scroll, pointer), dim=-1)
        return self.action_encoder(features)

    def imagine(self, belief: Belief, action: torch.Tensor,
                target: torch.Tensor) -> Prediction:
        """One recursive prediction. No actual future observations accepted."""
        slots = belief.latent.shape[1]
        target_flag = (torch.arange(slots, device=target.device)[None, :]
                       == target[:, None]).to(belief.latent.dtype).unsqueeze(-1)
        action_rows = action.unsqueeze(1).expand(-1, slots, -1)
        context = belief.context.unsqueeze(1).expand(-1, slots, -1)
        structured = torch.cat(
            (
                belief.present.unsqueeze(-1),
                belief.roles,
                belief.states,
                belief.flags,
                belief.boxes,
                belief.affordances,
            ),
            dim=-1,
        )
        conditioned_latent = (
            belief.latent + self.structured_feedback(structured)
        )
        features = torch.cat(
            (conditioned_latent, context, action_rows, target_flag), -1
        )
        h = functional.silu(self.condition(features))
        h = belief.latent + self.gated_projection(
            h * self.action_gate(action_rows)
        )
        for block in self.dynamics_blocks:
            h = block(h, belief.present, belief.parents)
        latent = self.next_norm(belief.latent + self.delta(h))
        present_logits = self.present_head(latent).squeeze(-1)
        one_to_zero, zero_to_one = self.flip_head(latent).chunk(2, dim=-1)
        birth = self.birth_state_head(latent)
        # Total probability for Boolean state transitions; birth is separate.
        existing = ((1.0 - belief.states) * zero_to_one.sigmoid()
                    + belief.states * (1.0 - one_to_zero.sigmoid()))
        old = belief.present.unsqueeze(-1)
        states = old * existing + (1.0 - old) * birth.sigmoid()
        role_logits = self.role_head(latent)
        flag_logits = self.flag_head(latent)
        affordance_logits = self.affordance_head(latent)
        pair_logits = (self.parent_query(latent)
                       @ self.parent_key(latent).transpose(-1, -2)) / 8.0
        pair_logits = pair_logits + present_logits.sigmoid().clamp_min(1e-4).log().unsqueeze(1)
        diagonal = torch.eye(slots, dtype=torch.bool, device=latent.device)
        pair_logits = pair_logits.masked_fill(diagonal, -1e4)
        parent_logits = torch.cat((self.root_head(latent), pair_logits), -1)
        batch_size = latent.shape[0]
        content_q = self.content_query(latent).reshape(
            batch_size, slots, 2, 64
        )
        content_k = self.content_key(belief.latent).reshape(
            batch_size, 2 * slots, 64
        )
        content_ptr = torch.einsum('bnfk,bmk->bnfm', content_q, content_k) / 8.0
        content_fix = self.content_fixed(latent).reshape(
            batch_size, slots, 2, 5
        )
        content_source_logits = torch.cat((content_fix, content_ptr), dim=-1)
        next_belief = Belief(
            latent, present_logits.sigmoid(), states, role_logits.softmax(-1),
            flag_logits.sigmoid(), self.box_head(latent),
            affordance_logits.sigmoid(), parent_logits.softmax(-1), belief.context,
        )
        return Prediction(
            next_belief, present_logits, one_to_zero, zero_to_one, birth,
            role_logits, flag_logits, affordance_logits, parent_logits,
            content_source_logits,
        )

    def correct_context(self, actual: Belief, predicted: Belief,
                        action: torch.Tensor,
                        known: torch.Tensor) -> Belief:
        """Update context only after a real transition; known [B,N] mask.

        The error compares aligned actual and predicted representations.
        This is a learned recurrent update, not a weight update.
        """
        weight = known.unsqueeze(-1).to(actual.latent.dtype)
        denominator = weight.sum(1).clamp_min(1.0)
        actual_mean = (actual.latent * weight).sum(1) / denominator
        error_mean = ((actual.latent - predicted.latent) * weight).sum(1) / denominator
        evidence = self.evidence(torch.cat((actual_mean, error_mean, action), -1))
        context = self.adaptation(evidence, actual.context)
        context = torch.where(known.any(1, keepdim=True), context, actual.context)
        return dataclasses.replace(actual, context=context)

    def evaluate_goal(self, belief: Belief, tokens: torch.Tensor,
                      token_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """tokens [B,L,384], token_mask [B,L] bool, at least one valid token.

        Returns goal logits [B,3,2] (three heads: satisfied, discounted return), proposal
        logits [B,N,4] and copy logits [B,N,2,L] (start/end).
        Mask invalid entities/actions/spans at the caller.
        """
        goal = self.goal_tokens(tokens)
        grounded, _ = self.goal_attention(
            belief.latent, goal, goal, key_padding_mask=~token_mask,
            need_weights=False,
        )
        context = belief.context.unsqueeze(1).expand(-1, belief.latent.shape[1], -1)
        features = self.goal_fusion(torch.cat((belief.latent, grounded,
            belief.states, belief.present.unsqueeze(-1), context), -1))
        weights = (self.goal_pool(features).squeeze(-1)
                   + belief.present.clamp_min(1e-4).log()).softmax(-1)
        pooled = (features * weights.unsqueeze(-1)).sum(1)
        queries = self.copy_query(features).reshape(
            features.shape[0], features.shape[1], 2, SLOT_DIM)
        copy_logits = torch.einsum('bncd,bld->bncl', queries, goal) / math.sqrt(SLOT_DIM)
        copy_logits = copy_logits.masked_fill(~token_mask[:, None, None, :], -1e4)
        goal_logits = torch.stack([head(pooled) for head in self.goal_heads], dim=1)
        return goal_logits, self.proposal_head(features), copy_logits

    def infer_action(self, before: Belief, after: Belief) -> tuple[torch.Tensor, torch.Tensor]:
        """Auxiliary inverse logits: kind [B,5], target [B,N+1].

        Target 0 means targetless; real slot i uses index i+1.
        Both beliefs must be aligned; ambiguity is expected.
        """
        features = self.inverse_features(torch.cat((before.latent, after.latent), -1))
        weights = before.present.unsqueeze(-1)
        pooled = (features * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        target = self.inverse_target(features).squeeze(-1)
        target = target.masked_fill(before.present < 0.5, -1e4)
        target = torch.cat((self.inverse_targetless(pooled), target), -1)
        return self.inverse_kind(pooled), target
