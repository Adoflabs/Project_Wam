"""Model-driven candidate construction, prediction, and ranking for WAM2."""

from __future__ import annotations

import dataclasses
import math

import torch

from browser_world.interaction import executor
from browser_world.wam2 import model
from browser_world.wam2 import observation

_PROPOSAL_PRIMITIVES = (
    executor.CLICK,
    executor.TYPE,
    executor.SELECT,
    executor.CHECK,
)
_DECISION_SAT_WEIGHT = 0.5
_DECISION_RET_WEIGHT = 0.5
_DECISION_DISAGREEMENT_WEIGHT = 0.25


@dataclasses.dataclass(frozen=True)
class CandidateAction:
    """An intervention grounded in the current observation.

    Attributes:
      candidate_id: Ephemeral index within one decision step.
      action: Actual Executor action to perform if selected.
      target_slot: Persistent WAM slot index, or None for targetless actions.
      instruction_span: Exact (start, end) character offsets in instruction.
      proposal_logit: Raw neural proposal output, or None if unsupported.
    """

    candidate_id: int
    action: executor.Action
    target_slot: int | None
    instruction_span: tuple[int, int] | None
    proposal_logit: float | None


@dataclasses.dataclass(frozen=True)
class CandidatePrediction:
    """A proposed action and its model-predicted consequence.

    Attributes:
      candidate: The grounded candidate action.
      prediction: One-step imagined future from WorldActionModel.imagine.
      satisfaction_score: Mean goal-satisfaction probability, or None.
      return_score: Mean discounted-return score, or None.
      disagreement: Std-dev across return heads (exploratory), or None.
      decision_score: Combined ranking score when heads are trained, or None.
    """

    candidate: CandidateAction
    prediction: model.Prediction
    satisfaction_score: float | None
    return_score: float | None
    disagreement: float | None
    decision_score: float | None


def decode_copy_spans(
    instruction: str,
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    token_offsets: torch.Tensor,
    token_mask: torch.Tensor,
    max_tokens: int,
    top_k: int,
) -> tuple[tuple[int, int], ...]:
    """Returns ranked valid character spans from copy logits.

    Args:
      instruction: Original natural-language instruction string.
      start_logits: Copy start logits of shape [L].
      end_logits: Copy end logits of shape [L].
      token_offsets: Token character offsets of shape [L, 2].
      token_mask: Boolean mask of valid non-special tokens of shape [L].
      max_tokens: Maximum token span length (>= 1).
      top_k: Maximum number of unique character spans to return (>= 1).

    Returns:
      Up to `top_k` deduplicated `(char_start, char_end)` spans ordered by
      descending `start_logits[s] + end_logits[e]`.
    """
    if start_logits.ndim != 1 or end_logits.ndim != 1:
        raise ValueError('start_logits and end_logits must be 1D [L].')
    seq_len = start_logits.shape[0]
    if end_logits.shape[0] != seq_len:
        raise ValueError('start_logits and end_logits length mismatch.')
    if token_offsets.shape != (seq_len, 2):
        raise ValueError('token_offsets must have shape [L, 2].')
    if token_mask.shape != (seq_len,):
        raise ValueError('token_mask must have shape [L].')
    if max_tokens < 1 or top_k < 1:
        raise ValueError('max_tokens and top_k must be >= 1.')

    mask_cpu = token_mask.detach().to(device='cpu', dtype=torch.bool).tolist()
    offsets_cpu = token_offsets.detach().to(device='cpu').tolist()
    starts_cpu = start_logits.detach().to(device='cpu').tolist()
    ends_cpu = end_logits.detach().to(device='cpu').tolist()

    scored_spans: list[tuple[float, int, int]] = []
    inst_len = len(instruction)
    for start_tok in range(seq_len):
        if not mask_cpu[start_tok]:
            continue
        char_start = int(offsets_cpu[start_tok][0])
        if char_start < 0 or char_start >= inst_len:
            continue
        max_end = min(seq_len, start_tok + max_tokens)
        for end_tok in range(start_tok, max_end):
            if not mask_cpu[end_tok]:
                continue
            char_end = int(offsets_cpu[end_tok][1])
            if char_end <= char_start or char_end > inst_len:
                continue
            score = float(starts_cpu[start_tok] + ends_cpu[end_tok])
            if math.isfinite(score):
                scored_spans.append((score, char_start, char_end))

    scored_spans.sort(key=lambda item: item[0], reverse=True)
    unique_spans: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for _, char_start, char_end in scored_spans:
        span = (char_start, char_end)
        if span in seen:
            continue
        seen.add(span)
        unique_spans.append(span)
        if len(unique_spans) >= top_k:
            break
    return tuple(unique_spans)


def _node_to_slot_map(
    aligned: observation.AlignedObservation,
) -> dict[int, int]:
    """Maps observed element node_ids to their aligned persistent slots."""
    return {
        elem.node_id: aligned.inputs.element_to_slot[idx]
        for idx, elem in enumerate(aligned.observation.elements)
    }


def proposal_score_for_affordance(
    proposal_logits: torch.Tensor,
    node_to_slot: dict[int, int],
    affordance: executor.Affordance,
) -> float | None:
    """Returns neural proposal logit for an affordance, or None if unsupported.

    Args:
      proposal_logits: Proposal logits of shape [1, N, 4].
      node_to_slot: Mapping from observed node_id to slot index.
      affordance: Legal observed affordance.

    Returns:
      Scalar float logit for supported element primitives, or None for scroll
      or unmapped targets.
    """
    if affordance.kind not in _PROPOSAL_PRIMITIVES:
        return None
    if affordance.target is None:
        return None
    slot = node_to_slot.get(affordance.target)
    if slot is None:
        return None
    prim_idx = _PROPOSAL_PRIMITIVES.index(affordance.kind)
    val = float(proposal_logits[0, slot, prim_idx].item())
    return val if math.isfinite(val) else None


def enumerate_candidates(
    aligned: observation.AlignedObservation,
    instruction: str,
    copy_spans: tuple[tuple[int, int], ...],
    available_actions: tuple[executor.Affordance, ...],
    proposal_logits: torch.Tensor | None = None,
) -> tuple[CandidateAction, ...]:
    """Builds candidates from observed legal interventions.

    Args:
      aligned: Current aligned browser observation.
      instruction: Exact user instruction string.
      copy_spans: Valid `(start, end)` character spans for typing actions.
      available_actions: Legal affordances from `executor.available_actions`.
      proposal_logits: Optional `[1, N, 4]` proposal logits from
        `evaluate_goal`.

    Returns:
      Tuple of grounded `CandidateAction` records.
    """
    inst_len = len(instruction)
    valid_spans: list[tuple[int, int]] = []
    seen_spans: set[tuple[int, int]] = set()
    for span in copy_spans:
        start, end = span
        if 0 <= start < end <= inst_len and span not in seen_spans:
            seen_spans.add(span)
            valid_spans.append(span)

    node_to_slot = _node_to_slot_map(aligned)
    candidates: list[CandidateAction] = []
    next_id = 0

    for aff in available_actions:
        if aff.target is None:
            target_slot: int | None = None
        else:
            target_slot = node_to_slot.get(aff.target)
            if target_slot is None:
                continue

        prop_logit = (
            proposal_score_for_affordance(proposal_logits, node_to_slot, aff)
            if proposal_logits is not None
            else None
        )

        if aff.kind == executor.TYPE:
            for span in valid_spans:
                literal = instruction[span[0]:span[1]]
                candidates.append(
                    CandidateAction(
                        candidate_id=next_id,
                        action=aff.action(text=literal),
                        target_slot=target_slot,
                        instruction_span=span,
                        proposal_logit=prop_logit,
                    )
                )
                next_id += 1
        else:
            candidates.append(
                CandidateAction(
                    candidate_id=next_id,
                    action=aff.action(),
                    target_slot=target_slot,
                    instruction_span=None,
                    proposal_logit=prop_logit,
                )
            )
            next_id += 1

    return tuple(candidates)


def _expand_belief(belief: model.Belief, batch_size: int) -> model.Belief:
    """Expands a batch-1 belief along dim 0 without copying memory."""
    return model.Belief(
        latent=belief.latent.expand(batch_size, -1, -1),
        present=belief.present.expand(batch_size, -1),
        states=belief.states.expand(batch_size, -1, -1),
        roles=belief.roles.expand(batch_size, -1, -1),
        flags=belief.flags.expand(batch_size, -1, -1),
        boxes=belief.boxes.expand(batch_size, -1, -1),
        affordances=belief.affordances.expand(batch_size, -1, -1),
        parents=belief.parents.expand(batch_size, -1, -1),
        context=belief.context.expand(batch_size, -1),
    )


def _slice_prediction(pred: model.Prediction, idx: int) -> model.Prediction:
    """Extracts a batch-1 Prediction at index `idx`."""
    b = pred.belief
    single_belief = model.Belief(
        latent=b.latent[idx:idx + 1],
        present=b.present[idx:idx + 1],
        states=b.states[idx:idx + 1],
        roles=b.roles[idx:idx + 1],
        flags=b.flags[idx:idx + 1],
        boxes=b.boxes[idx:idx + 1],
        affordances=b.affordances[idx:idx + 1],
        parents=b.parents[idx:idx + 1],
        context=b.context[idx:idx + 1],
    )
    return model.Prediction(
        belief=single_belief,
        present_logits=pred.present_logits[idx:idx + 1],
        one_to_zero_logits=pred.one_to_zero_logits[idx:idx + 1],
        zero_to_one_logits=pred.zero_to_one_logits[idx:idx + 1],
        birth_state_logits=pred.birth_state_logits[idx:idx + 1],
        role_logits=pred.role_logits[idx:idx + 1],
        flag_logits=pred.flag_logits[idx:idx + 1],
        affordance_logits=pred.affordance_logits[idx:idx + 1],
        parent_logits=pred.parent_logits[idx:idx + 1],
        content_source_logits=pred.content_source_logits[idx:idx + 1],
    )


def predict_candidates(
    world: model.WorldActionModel,
    belief: model.Belief,
    candidates: tuple[CandidateAction, ...],
    aligned: observation.AlignedObservation,
    encoder: observation.TextEncoder,
    *,
    batch_size: int = 16,
    pointer: tuple[float, float] = (0.0, 0.0),
) -> tuple[CandidatePrediction, ...]:
    """Imagines action consequences without interacting with the page.

    Args:
      world: WorldActionModel instance.
      belief: Current actual belief of batch size 1.
      candidates: Grounded candidate actions to imagine.
      aligned: Pre-action aligned observation (for action encoding only).
      encoder: Text encoder for action text features.
      batch_size: Maximum candidates per forward chunk (<= 16 recommended).
      pointer: Current viewport pointer coordinates `(x, y)`.

    Returns:
      Tuple of `CandidatePrediction` aligned 1-to-1 with `candidates`, with
      goal/decision scores initialized to None.
    """
    if batch_size < 1:
        raise ValueError(f'batch_size must be >= 1, got {batch_size}.')
    if not candidates:
        return ()

    world.eval()
    results: list[CandidatePrediction] = []
    with torch.inference_mode():
        encoded = [
            observation.encode_action(
                cand.action, aligned, encoder, pointer
            )
            for cand in candidates
        ]
        for start in range(0, len(candidates), batch_size):
            chunk_cands = candidates[start:start + batch_size]
            chunk_enc = encoded[start:start + batch_size]
            n = len(chunk_cands)

            belief_batch = _expand_belief(belief, n)
            kind = torch.cat([e.kind for e in chunk_enc], dim=0)
            target = torch.cat([e.target for e in chunk_enc], dim=0)
            text = torch.cat([e.text for e in chunk_enc], dim=0)
            checked = torch.cat([e.checked for e in chunk_enc], dim=0)
            scroll = torch.cat([e.scroll for e in chunk_enc], dim=0)
            ptr = torch.cat([e.pointer for e in chunk_enc], dim=0)

            act_latent = world.encode_action(
                belief_batch, kind, target, text, checked, scroll, ptr
            )
            pred_batch = world.imagine(belief_batch, act_latent, target)
            for i, cand in enumerate(chunk_cands):
                results.append(
                    CandidatePrediction(
                        candidate=cand,
                        prediction=_slice_prediction(pred_batch, i),
                        satisfaction_score=None,
                        return_score=None,
                        disagreement=None,
                        decision_score=None,
                    )
                )
    return tuple(results)


def score_predictions(
    world: model.WorldActionModel,
    predictions: tuple[CandidatePrediction, ...],
    instruction_tokens: torch.Tensor,
    instruction_mask: torch.Tensor,
    *,
    satisfaction_trained: bool,
    return_trained: bool,
    batch_size: int = 16,
) -> tuple[CandidatePrediction, ...]:
    """Evaluates imagined futures with `world.evaluate_goal` when trained.

    Untrained heads are never used to rank futures: if `satisfaction_trained` is
    False (and `return_trained` is False), predictions are returned unchanged
    with `None` scores.

    Args:
      world: WorldActionModel instance.
      predictions: Candidate predictions from `predict_candidates`.
      instruction_tokens: Encoded instruction tokens `[1, L, 384]`.
      instruction_mask: Valid token mask `[1, L]`.
      satisfaction_trained: Whether goal satisfaction supervision is available.
      return_trained: Whether discounted return supervision is available.
      batch_size: Evaluation chunk size.

    Returns:
      Updated tuple of `CandidatePrediction`.
    """
    if not predictions or (not satisfaction_trained and not return_trained):
        return predictions

    world.eval()
    scored: list[CandidatePrediction] = []
    with torch.inference_mode():
        for start in range(0, len(predictions), batch_size):
            chunk = predictions[start:start + batch_size]
            n = len(chunk)
            belief_batch = model.Belief(
                latent=torch.cat(
                    [p.prediction.belief.latent for p in chunk], 0
                ),
                present=torch.cat(
                    [p.prediction.belief.present for p in chunk], 0
                ),
                states=torch.cat(
                    [p.prediction.belief.states for p in chunk], 0
                ),
                roles=torch.cat([p.prediction.belief.roles for p in chunk], 0),
                flags=torch.cat([p.prediction.belief.flags for p in chunk], 0),
                boxes=torch.cat([p.prediction.belief.boxes for p in chunk], 0),
                affordances=torch.cat(
                    [p.prediction.belief.affordances for p in chunk], 0
                ),
                parents=torch.cat(
                    [p.prediction.belief.parents for p in chunk], 0
                ),
                context=torch.cat(
                    [p.prediction.belief.context for p in chunk], 0
                ),
            )
            tok_batch = instruction_tokens.expand(n, -1, -1)
            mask_batch = instruction_mask.expand(n, -1)
            goal_logits, _, _ = world.evaluate_goal(
                belief_batch, tok_batch, mask_batch
            )
            probs = goal_logits.sigmoid()  # [n, 3, 2]: (satisfied, return)
            sat_probs = probs[:, :, 0]
            ret_probs = probs[:, :, 1]

            for i, pred in enumerate(chunk):
                sat_score = (
                    float(sat_probs[i].mean().item())
                    if satisfaction_trained
                    else None
                )
                ret_score = (
                    float(ret_probs[i].mean().item())
                    if return_trained
                    else None
                )
                disagree = (
                    float(ret_probs[i].std(unbiased=False).item())
                    if return_trained
                    else None
                )
                if satisfaction_trained and return_trained:
                    assert sat_score is not None and ret_score is not None
                    assert disagree is not None
                    d_score = (
                        _DECISION_SAT_WEIGHT * sat_score
                        + _DECISION_RET_WEIGHT * ret_score
                        - _DECISION_DISAGREEMENT_WEIGHT * disagree
                    )
                elif satisfaction_trained:
                    d_score = sat_score
                else:
                    d_score = None

                scored.append(
                    dataclasses.replace(
                        pred,
                        satisfaction_score=sat_score,
                        return_score=ret_score,
                        disagreement=disagree,
                        decision_score=d_score,
                    )
                )
    return tuple(scored)


def select_candidate(
    predictions: tuple[CandidatePrediction, ...],
    *,
    allow_proposal_only: bool = False,
) -> CandidatePrediction | None:
    """Selects the highest supported model-scored candidate.

    Args:
      predictions: Evaluated candidate predictions.
      allow_proposal_only: If True and no `decision_score` is present, fall back
        to the highest finite `candidate.proposal_logit`.

    Returns:
      The highest-scoring `CandidatePrediction`, or None if no supported score
      exists.
    """
    value_scored = [
        p for p in predictions
        if p.decision_score is not None and math.isfinite(p.decision_score)
    ]
    if value_scored:
        return max(value_scored, key=lambda p: float(p.decision_score or 0.0))

    if allow_proposal_only:
        prop_scored = [
            p for p in predictions
            if (
                p.candidate.proposal_logit is not None
                and math.isfinite(p.candidate.proposal_logit)
            )
        ]
        if prop_scored:
            return max(
                prop_scored,
                key=lambda p: float(p.candidate.proposal_logit or 0.0),
            )
    return None


def resolve_candidate_action(
    candidate: CandidateAction,
    aligned: observation.AlignedObservation,
) -> executor.Action | None:
    """Resolves a candidate's persistent slot to an observed Executor action.

    Rejects imagined-only or disappeared slots.
    """
    if candidate.action.kind == executor.SCROLL:
        return candidate.action
    if candidate.target_slot is None:
        return None
    if (
        candidate.target_slot < 0
        or candidate.target_slot >= len(aligned.inputs.slot_to_element)
    ):
        return None
    elem_idx = aligned.inputs.slot_to_element[candidate.target_slot]
    if elem_idx is None:
        return None
    node_id = aligned.observation.elements[elem_idx].node_id
    return dataclasses.replace(candidate.action, target=node_id)
