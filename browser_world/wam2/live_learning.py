"""Online experience-learning, replay retention, and checkpointing for WAM2."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
import math
import os
import pathlib
import random
from typing import Any

import torch
from torch.nn import functional

from browser_world.interaction import executor
from browser_world.wam2 import content
from browser_world.wam2 import losses
from browser_world.wam2 import model
from browser_world.wam2 import observation
from browser_world.wam2 import planning
from browser_world.wam2 import training_data

TERMINAL_OUTCOMES = ('success', 'failure', 'incomplete')
RETURN_DISCOUNT = 0.97

LIVE_LOSS_WEIGHTS = {
    'existence': 1.0,
    'directional_states': 2.0,
    'affordances': 1.0,
    'content_source': 1.0,
    'proposal': 1.0,
    'pairwise_ranking': 0.5,
    'copy': 1.0,
    'goal_satisfaction': 1.0,
    'return': 0.5,
}

DYNAMICS_HEAD_MODULES = (
    'present_head',
    'flip_head',
    'birth_state_head',
    'affordance_head',
    'content_query',
    'content_key',
    'content_fixed',
)

DECISION_HEAD_MODULES = (
    'goal_tokens',
    'goal_attention',
    'goal_fusion',
    'goal_pool',
    'goal_heads',
    'proposal_head',
    'copy_query',
)

TRAINABLE_MODULE_PREFIXES = DYNAMICS_HEAD_MODULES + DECISION_HEAD_MODULES

_PROPOSAL_PRIMITIVES = (
    executor.CLICK,
    executor.TYPE,
    executor.SELECT,
    executor.CHECK,
)


@dataclasses.dataclass(frozen=True)
class LearningConfig:
    """Conservative initial settings for bounded online learning."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    replay_examples: int = 4
    updates_per_feedback: int = 1
    max_live_examples: int = 256
    max_copy_tokens: int = 32
    seed: int = 0
    reference_degradation_tolerance: float = 0.10
    reference_absolute_eps: float = 1e-4


@dataclasses.dataclass(frozen=True)
class ActionFeedback:
    """Explicit supervision provided during an interaction."""

    decision_id: str
    approved_candidate_id: int | None = None
    rejected_candidate_ids: tuple[int, ...] = ()
    literal_span: tuple[int, int] | None = None
    goal_satisfied: bool | None = None
    constraint_violated: bool | None = None
    terminal_outcome: str | None = None


@dataclasses.dataclass(frozen=True)
class LearningSample:
    """Detached transition and feedback data for one supervised step."""

    sample_id: str
    decision_id: str
    before_aligned: observation.AlignedObservation
    after_aligned: observation.AlignedObservation | None
    executed_candidate: planning.CandidateAction | None
    candidates: tuple[planning.CandidateAction, ...]
    pointer: tuple[float, float]
    instruction: str
    instruction_tokens: torch.Tensor
    instruction_mask: torch.Tensor
    token_offsets: torch.Tensor
    feedback: ActionFeedback | None = None
    previous_belief: model.Belief | None = None
    context: torch.Tensor | None = None


@dataclasses.dataclass(frozen=True)
class UpdateReport:
    """Measured telemetry from one attempted optimizer update."""

    learning_sample_id: str
    status: str
    supervised_objectives: tuple[str, ...]
    raw_losses: dict[str, float]
    loss_before: float
    loss_after: float
    gradient_norm: float
    changed_parameter_tensors: int
    parameter_update_norm: float
    model_revision_before: int
    model_revision_after: int
    reference_loss_before: float | None
    reference_loss_after: float | None
    rejection_reason: str | None = None


def validate_feedback(
    feedback: ActionFeedback,
    *,
    valid_candidate_ids: set[int],
    instruction_length: int,
) -> None:
    """Validates an ActionFeedback record against the current decision.

    Raises:
      ValueError: If candidate references, span, or terminal outcome are
        invalid.
    """
    if (
        feedback.approved_candidate_id is not None
        and feedback.approved_candidate_id not in valid_candidate_ids
    ):
        raise ValueError(
            f'Unknown approved_candidate_id {feedback.approved_candidate_id}.'
        )
    for cid in feedback.rejected_candidate_ids:
        if cid not in valid_candidate_ids:
            raise ValueError(f'Unknown rejected_candidate_id {cid}.')
    if (
        feedback.approved_candidate_id is not None
        and feedback.approved_candidate_id in feedback.rejected_candidate_ids
    ):
        raise ValueError(
            'Approved and rejected candidate IDs cannot overlap.'
        )
    if feedback.literal_span is not None:
        start, end = feedback.literal_span
        if start < 0 or end <= start or end > instruction_length:
            raise ValueError(
                f'Invalid literal_span {feedback.literal_span} for '
                f'instruction length {instruction_length}.'
            )
    if (
        feedback.terminal_outcome is not None
        and feedback.terminal_outcome not in TERMINAL_OUTCOMES
    ):
        raise ValueError(
            f'Unsupported terminal_outcome {feedback.terminal_outcome!r}.'
        )


def configure_trainable_parameters(
    world: model.WorldActionModel,
) -> tuple[list[torch.nn.Parameter], tuple[str, ...], tuple[str, ...]]:
    """Freezes backbone/perception/GRU and enables only head-level updates.

    Returns:
      Tuple `(trainable_params, updated_names, frozen_names)`.
    """
    trainable_params: list[torch.nn.Parameter] = []
    updated_names: list[str] = []
    frozen_names: list[str] = []

    for name, param in world.named_parameters():
        prefix = name.split('.', 1)[0]
        if prefix in TRAINABLE_MODULE_PREFIXES:
            param.requires_grad_(True)
            trainable_params.append(param)
            updated_names.append(name)
        else:
            param.requires_grad_(False)
            frozen_names.append(name)
    return trainable_params, tuple(updated_names), tuple(frozen_names)


def _detach_belief(belief: model.Belief | None) -> model.Belief | None:
    """Returns a detached copy of a Belief so no graph is retained."""
    if belief is None:
        return None
    return model.Belief(
        latent=belief.latent.detach(),
        present=belief.present.detach(),
        states=belief.states.detach(),
        roles=belief.roles.detach(),
        flags=belief.flags.detach(),
        boxes=belief.boxes.detach(),
        affordances=belief.affordances.detach(),
        parents=belief.parents.detach(),
        context=belief.context.detach(),
    )


def _with_context(
    belief: model.Belief,
    context: torch.Tensor | None,
) -> model.Belief:
    """Applies a detached context tensor to a Belief if provided."""
    if context is None:
        return belief
    return dataclasses.replace(belief, context=context.detach())


def _dynamics_loss_for_transition(
    world: model.WorldActionModel,
    before_aligned: observation.AlignedObservation,
    after_aligned: observation.AlignedObservation,
    action: executor.Action,
    pointer: tuple[float, float],
    encoder: observation.TextEncoder,
    *,
    previous_belief: model.Belief | None = None,
    context: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Computes directly supervised head-level dynamics and content losses."""
    before_belief = observation.observe(
        world, before_aligned, previous=_detach_belief(previous_belief)
    )
    before_belief = _with_context(before_belief, context)
    act_in = observation.encode_action(action, before_aligned, encoder, pointer)
    act_lat = world.encode_action(
        before_belief,
        act_in.kind,
        act_in.target,
        act_in.text,
        act_in.checked,
        act_in.scroll,
        act_in.pointer,
    )
    pred = world.imagine(before_belief, act_lat, act_in.target)

    ex_loss, _ = losses.existence_loss(
        pred, before_aligned.inputs.present, after_aligned.inputs
    )
    dir_loss, _ = losses.directional_state_loss(
        pred,
        before_aligned.inputs.states,
        before_aligned.inputs.present,
        after_aligned.inputs,
    )
    present_mask = (
        (after_aligned.inputs.present > 0.5)
        & after_aligned.inputs.existence_known
    )
    aff_bce = functional.binary_cross_entropy_with_logits(
        pred.affordance_logits,
        after_aligned.inputs.affordances,
        reduction='none',
    )
    aff_loss = losses.masked_mean(aff_bce, present_mask.unsqueeze(-1))

    result: dict[str, torch.Tensor] = {
        'existence': ex_loss,
        'directional_states': dir_loss,
        'affordances': aff_loss,
    }

    if before_aligned.exact_content and after_aligned.exact_content:
        legal = content.legal_source_mask(
            before_aligned.exact_content,
            before_aligned.inputs.present,
            action.text,
            device=pred.content_source_logits.device,
        )
        targets, eligible = content.build_content_targets(
            before_aligned.exact_content,
            before_aligned.inputs.present,
            after_aligned.exact_content,
            after_aligned.inputs.present,
            after_aligned.inputs.existence_known,
            action.text,
            device=pred.content_source_logits.device,
        )
        if bool(eligible.any().item()):
            result['content_source'] = losses.set_valued_content_loss(
                pred.content_source_logits, legal, targets, eligible
            )
    return result


def _copy_span_loss(
    instruction: str,
    literal_span: tuple[int, int],
    target_slot: int,
    copy_logits: torch.Tensor,
    token_offsets: torch.Tensor,
    token_mask: torch.Tensor,
    max_tokens: int,
) -> torch.Tensor | None:
    """Computes set-valued copy loss for exact round-trip token spans."""
    char_start, char_end = literal_span
    literal = instruction[char_start:char_end]
    if not literal:
        return None

    seq_len = copy_logits.shape[-1]
    mask_cpu = token_mask[0].detach().to('cpu', dtype=torch.bool).tolist()
    offsets_cpu = token_offsets.detach().to('cpu').tolist()
    start_l = copy_logits[0, target_slot, 0, :]
    end_l = copy_logits[0, target_slot, 1, :]

    all_scores: list[torch.Tensor] = []
    correct_scores: list[torch.Tensor] = []
    inst_len = len(instruction)
    for s_tok in range(seq_len):
        if not mask_cpu[s_tok]:
            continue
        c_s = int(offsets_cpu[s_tok][0])
        if c_s < 0 or c_s >= inst_len:
            continue
        max_e = min(seq_len, s_tok + max_tokens)
        for e_tok in range(s_tok, max_e):
            if not mask_cpu[e_tok]:
                continue
            c_e = int(offsets_cpu[e_tok][1])
            if c_e <= c_s or c_e > inst_len:
                continue
            score = start_l[s_tok] + end_l[e_tok]
            all_scores.append(score)
            if instruction[c_s:c_e] == literal:
                correct_scores.append(score)

    if not all_scores or not correct_scores:
        return None
    all_t = torch.stack(all_scores)
    cor_t = torch.stack(correct_scores)
    return torch.logsumexp(all_t, dim=0) - torch.logsumexp(cor_t, dim=0)


def _proposal_logit_tensor(
    proposal_logits: torch.Tensor,
    cand: planning.CandidateAction,
) -> torch.Tensor | None:
    """Indexes the scalar proposal logit tensor for `cand`, if supported."""
    if cand.target_slot is None or cand.action.kind not in _PROPOSAL_PRIMITIVES:
        return None
    prim_idx = _PROPOSAL_PRIMITIVES.index(cand.action.kind)
    return proposal_logits[0, cand.target_slot, prim_idx]


def _decision_losses_for_sample(
    world: model.WorldActionModel,
    sample: LearningSample,
    max_copy_tokens: int,
) -> dict[str, torch.Tensor]:
    """Computes proposal, pairwise ranking, copy, and goal satisfaction loss."""
    feedback = sample.feedback
    if feedback is None:
        return {}

    before_belief = observation.observe(
        world,
        sample.before_aligned,
        previous=_detach_belief(sample.previous_belief),
    )
    before_belief = _with_context(before_belief, sample.context)
    _, proposal_logits, copy_logits = world.evaluate_goal(
        before_belief, sample.instruction_tokens, sample.instruction_mask
    )

    cand_by_id = {c.candidate_id: c for c in sample.candidates}
    raw: dict[str, torch.Tensor] = {}

    sup_logits: list[torch.Tensor] = []
    sup_targets: list[float] = []
    preferred_logit: torch.Tensor | None = None
    if feedback.approved_candidate_id is not None:
        app_cand = cand_by_id.get(feedback.approved_candidate_id)
        if app_cand is not None:
            logit = _proposal_logit_tensor(proposal_logits, app_cand)
            if logit is not None:
                preferred_logit = logit
                sup_logits.append(logit)
                sup_targets.append(1.0)

    rejected_logits: list[torch.Tensor] = []
    for rej_id in feedback.rejected_candidate_ids:
        rej_cand = cand_by_id.get(rej_id)
        if rej_cand is None:
            continue
        logit = _proposal_logit_tensor(proposal_logits, rej_cand)
        if logit is not None:
            rejected_logits.append(logit)
            sup_logits.append(logit)
            sup_targets.append(0.0)

    if sup_logits:
        l_t = torch.stack(sup_logits)
        t_t = torch.tensor(sup_targets, dtype=l_t.dtype, device=l_t.device)
        raw['proposal'] = functional.binary_cross_entropy_with_logits(l_t, t_t)

    if preferred_logit is not None and rejected_logits:
        rej_t = torch.stack(rejected_logits)
        raw['pairwise_ranking'] = functional.softplus(
            rej_t - preferred_logit
        ).mean()

    if feedback.approved_candidate_id is not None:
        app_cand = cand_by_id.get(feedback.approved_candidate_id)
        span = feedback.literal_span or (
            app_cand.instruction_span if app_cand is not None else None
        )
        if (
            app_cand is not None
            and app_cand.action.kind == executor.TYPE
            and app_cand.target_slot is not None
            and span is not None
        ):
            copy_l = _copy_span_loss(
                sample.instruction,
                span,
                app_cand.target_slot,
                copy_logits,
                sample.token_offsets,
                sample.instruction_mask,
                max_copy_tokens,
            )
            if copy_l is not None:
                raw['copy'] = copy_l

    if feedback.goal_satisfied is not None:
        eval_aligned = sample.after_aligned or sample.before_aligned
        eval_prev = sample.before_aligned if sample.after_aligned else None
        eval_prev_belief = (
            observation.observe(
                world,
                eval_prev,
                previous=_detach_belief(sample.previous_belief),
            )
            if eval_prev is not None
            else _detach_belief(sample.previous_belief)
        )
        eval_belief = observation.observe(
            world, eval_aligned, previous=eval_prev_belief
        )
        eval_belief = _with_context(eval_belief, sample.context)
        goal_logits, _, _ = world.evaluate_goal(
            eval_belief, sample.instruction_tokens, sample.instruction_mask
        )
        sat_logits = goal_logits[:, :, 0]
        target_val = 1.0 if feedback.goal_satisfied else 0.0
        sat_target = torch.full_like(sat_logits, target_val)
        raw['goal_satisfaction'] = functional.binary_cross_entropy_with_logits(
            sat_logits, sat_target
        )

    return raw


def return_targets_for_episode(
    step_count: int,
    terminal_outcome: str | None,
    discount: float = RETURN_DISCOUNT,
) -> tuple[float, ...] | None:
    """Computes discounted return targets for a completed episode.

    Incomplete or None outcomes return None so no false return target is used.
    """
    if step_count < 1 or terminal_outcome not in ('success', 'failure'):
        return None
    terminal_reward = 1.0 if terminal_outcome == 'success' else 0.0
    last_idx = step_count - 1
    return tuple(
        terminal_reward * (discount ** (last_idx - t))
        for t in range(step_count)
    )


class LiveLearner:
    """Learns from verified transitions and explicit human supervision."""

    def __init__(
        self,
        world: model.WorldActionModel,
        encoder: observation.TextEncoder,
        role_to_index: dict[str, int],
        config: LearningConfig,
        *,
        base_checkpoint_sha256: str = '',
        replay_windows: Sequence[training_data.ExperienceWindow] = (),
        reference_windows: Sequence[training_data.ExperienceWindow] = (),
        model_revision: int = 0,
        supervision_counts: dict[str, int] | None = None,
    ) -> None:
        self.world = world
        self.encoder = encoder
        self.role_to_index = dict(role_to_index)
        self.config = config
        self.base_checkpoint_sha256 = base_checkpoint_sha256
        self.model_revision = model_revision
        self._rng = random.Random(config.seed)

        self._trainable_params, self.updated_parameter_groups, (
            self.frozen_parameter_groups
        ) = configure_trainable_parameters(self.world)
        self.optimizer = torch.optim.AdamW(
            self._trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.live_samples: list[LearningSample] = []
        self._replay_windows = list(replay_windows)
        self._reference_windows = list(reference_windows)
        self.supervision_counts: dict[str, int] = dict(
            supervision_counts
            or {
                'transitions': 0,
                'proposal': 0,
                'copy': 0,
                'goal_satisfaction': 0,
                'return': 0,
            }
        )
        self.metrics_history: list[dict[str, Any]] = []

    def capability_status(self) -> dict[str, Any]:
        """Returns explicit readiness flags based on observed supervision."""
        return {
            'dynamics_loaded': True,
            'teaching_ready': True,
            'proposal_supervised': (
                self.supervision_counts.get('proposal', 0) > 0
            ),
            'copy_supervised': self.supervision_counts.get('copy', 0) > 0,
            'satisfaction_supervised': (
                self.supervision_counts.get('goal_satisfaction', 0) > 0
            ),
            'return_supervised': self.supervision_counts.get('return', 0) > 0,
            'supervision_counts': dict(self.supervision_counts),
            'model_revision': self.model_revision,
        }

    def autonomous_block_reason(self) -> str | None:
        """Returns reason if autonomous execution is not supported."""
        if self.supervision_counts.get('proposal', 0) < 1:
            return (
                'Autonomous mode disabled: no supervised action proposals yet. '
                'Use Teach mode to approve or correct actions first.'
            )
        return None

    def _encode_window_step(
        self,
        win: training_data.ExperienceWindow,
    ) -> tuple[
        observation.AlignedObservation,
        observation.AlignedObservation,
        executor.Action,
        tuple[float, float],
    ]:
        """Aligns step 0 of an ExperienceWindow."""
        before_aligned = observation.encode_observation(
            win.observations[0],
            self.encoder,
            self.role_to_index,
            slot_count=self.world.slot_count,
            previous=None,
        )
        after_aligned = observation.encode_observation(
            win.observations[1],
            self.encoder,
            self.role_to_index,
            slot_count=self.world.slot_count,
            previous=before_aligned,
        )
        return before_aligned, after_aligned, win.actions[0], win.pointers[0]

    def _reference_dynamics_loss(self) -> float | None:
        """Evaluates dynamics head loss on the fixed reference cohort."""
        if not self._reference_windows:
            return None
        self.world.eval()
        total = 0.0
        with torch.no_grad():
            for win in self._reference_windows:
                b_al, a_al, act, ptr = self._encode_window_step(win)
                comp = _dynamics_loss_for_transition(
                    self.world, b_al, a_al, act, ptr, self.encoder
                )
                step_total = sum(
                    LIVE_LOSS_WEIGHTS[k] * v for k, v in comp.items()
                )
                total += float(step_total.item())
        return total / len(self._reference_windows)

    def _compute_sample_loss(
        self,
        sample: LearningSample,
        *,
        include_replay: bool,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes total weighted loss and raw scalar terms for `sample`."""
        raw_tensors: dict[str, torch.Tensor] = {}

        if (
            sample.after_aligned is not None
            and sample.executed_candidate is not None
        ):
            dyn = _dynamics_loss_for_transition(
                self.world,
                sample.before_aligned,
                sample.after_aligned,
                sample.executed_candidate.action,
                sample.pointer,
                self.encoder,
                previous_belief=sample.previous_belief,
                context=sample.context,
            )
            raw_tensors.update(dyn)

            if include_replay and self._replay_windows:
                k = min(self.config.replay_examples, len(self._replay_windows))
                replay_batch = self._rng.sample(self._replay_windows, k)
                replay_losses: list[torch.Tensor] = []
                for win in replay_batch:
                    b_al, a_al, act, ptr = self._encode_window_step(win)
                    r_comp = _dynamics_loss_for_transition(
                        self.world, b_al, a_al, act, ptr, self.encoder
                    )
                    replay_losses.append(
                        sum(LIVE_LOSS_WEIGHTS[n] * v for n, v in r_comp.items())
                    )
                raw_tensors['replay_dynamics'] = torch.stack(
                    replay_losses
                ).mean()

        dec = _decision_losses_for_sample(
            self.world, sample, self.config.max_copy_tokens
        )
        raw_tensors.update(dec)

        if not raw_tensors:
            zero = sum(p.sum() * 0.0 for p in self._trainable_params)
            return zero, {}

        total = sum(
            (
                v
                if k == 'replay_dynamics'
                else LIVE_LOSS_WEIGHTS[k] * v
            )
            for k, v in raw_tensors.items()
        )
        raw_floats = {k: float(v.item()) for k, v in raw_tensors.items()}
        return total, raw_floats

    def _snapshot_state(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Captures trainable parameter tensors and optimizer state."""
        param_snap = {
            name: param.detach().clone()
            for name, param in self.world.named_parameters()
            if param.requires_grad
        }
        opt_snap = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in self.optimizer.state_dict().items()
        }
        opt_snap = self.optimizer.state_dict()
        return param_snap, opt_snap

    def _restore_snapshot(
        self,
        param_snap: dict[str, torch.Tensor],
        opt_snap: dict[str, Any],
    ) -> None:
        """Restores trainable parameters and optimizer state."""
        with torch.no_grad():
            for name, param in self.world.named_parameters():
                if name in param_snap:
                    param.copy_(param_snap[name])
        self.optimizer.load_state_dict(opt_snap)

    def learn_from_sample(self, sample: LearningSample) -> UpdateReport:
        """Records `sample` and performs one bounded, verified update."""
        self.live_samples.append(sample)
        if len(self.live_samples) > self.config.max_live_examples:
            self.live_samples.pop(0)

        rev_before = self.model_revision
        self.world.eval()
        with torch.no_grad():
            loss_before_t, raw_before = self._compute_sample_loss(
                sample, include_replay=False
            )
        if not raw_before:
            return UpdateReport(
                learning_sample_id=sample.sample_id,
                status='no_supported_learning_signal',
                supervised_objectives=(),
                raw_losses={},
                loss_before=0.0,
                loss_after=0.0,
                gradient_norm=0.0,
                changed_parameter_tensors=0,
                parameter_update_norm=0.0,
                model_revision_before=rev_before,
                model_revision_after=rev_before,
                reference_loss_before=None,
                reference_loss_after=None,
            )

        loss_before = float(loss_before_t.item())
        has_dynamics = (
            sample.after_aligned is not None
            and sample.executed_candidate is not None
        )
        ref_before = (
            self._reference_dynamics_loss() if has_dynamics else None
        )
        param_snap, opt_snap = self._snapshot_state()

        try:
            self.world.train()
            self.optimizer.zero_grad(set_to_none=True)
            train_loss, raw_train = self._compute_sample_loss(
                sample, include_replay=True
            )
            if not torch.isfinite(train_loss).item():
                raise FloatingPointError('Non-finite live learning loss.')
            train_loss.backward()
            grad_norm_t = torch.nn.utils.clip_grad_norm_(
                self._trainable_params,
                self.config.gradient_clip,
                error_if_nonfinite=True,
            )
            grad_norm = float(grad_norm_t.item())
            self.optimizer.step()
            self.world.eval()
        except (FloatingPointError, RuntimeError) as exc:
            self._restore_snapshot(param_snap, opt_snap)
            self.world.eval()
            return UpdateReport(
                learning_sample_id=sample.sample_id,
                status='weight_update_rejected',
                supervised_objectives=tuple(raw_before.keys()),
                raw_losses=raw_before,
                loss_before=loss_before,
                loss_after=loss_before,
                gradient_norm=0.0,
                changed_parameter_tensors=0,
                parameter_update_norm=0.0,
                model_revision_before=rev_before,
                model_revision_after=rev_before,
                reference_loss_before=ref_before,
                reference_loss_after=ref_before,
                rejection_reason=str(exc),
            )

        for name, param in self.world.named_parameters():
            if param.requires_grad and not torch.isfinite(param).all().item():
                self._restore_snapshot(param_snap, opt_snap)
                return UpdateReport(
                    learning_sample_id=sample.sample_id,
                    status='weight_update_rejected',
                    supervised_objectives=tuple(raw_before.keys()),
                    raw_losses=raw_before,
                    loss_before=loss_before,
                    loss_after=loss_before,
                    gradient_norm=grad_norm,
                    changed_parameter_tensors=0,
                    parameter_update_norm=0.0,
                    model_revision_before=rev_before,
                    model_revision_after=rev_before,
                    reference_loss_before=ref_before,
                    reference_loss_after=ref_before,
                    rejection_reason=f'Non-finite parameter {name}.',
                )

        ref_after = self._reference_dynamics_loss() if has_dynamics else None
        if (
            ref_before is not None
            and ref_after is not None
            and ref_after
            > ref_before * (1.0 + self.config.reference_degradation_tolerance)
            + self.config.reference_absolute_eps
        ):
            self._restore_snapshot(param_snap, opt_snap)
            return UpdateReport(
                learning_sample_id=sample.sample_id,
                status='weight_update_rejected',
                supervised_objectives=tuple(raw_before.keys()),
                raw_losses=raw_before,
                loss_before=loss_before,
                loss_after=loss_before,
                gradient_norm=grad_norm,
                changed_parameter_tensors=0,
                parameter_update_norm=0.0,
                model_revision_before=rev_before,
                model_revision_after=rev_before,
                reference_loss_before=ref_before,
                reference_loss_after=ref_after,
                rejection_reason=(
                    f'Reference loss degraded from {ref_before:.4f} '
                    f'to {ref_after:.4f}.'
                ),
            )

        changed_tensors = 0
        sq_diff_sum = 0.0
        for name, param in self.world.named_parameters():
            if name in param_snap:
                diff = param.detach() - param_snap[name]
                norm_sq = float((diff * diff).sum().item())
                if norm_sq > 0.0:
                    changed_tensors += 1
                    sq_diff_sum += norm_sq
        update_norm = math.sqrt(sq_diff_sum)

        with torch.no_grad():
            loss_after_t, _ = self._compute_sample_loss(
                sample, include_replay=False
            )
        loss_after = float(loss_after_t.item())

        self.model_revision += 1
        if has_dynamics:
            self.supervision_counts['transitions'] += 1
        if 'proposal' in raw_train:
            self.supervision_counts['proposal'] += 1
        if 'copy' in raw_train:
            self.supervision_counts['copy'] += 1
        if 'goal_satisfaction' in raw_train:
            self.supervision_counts['goal_satisfaction'] += 1

        report = UpdateReport(
            learning_sample_id=sample.sample_id,
            status='weight_update_accepted',
            supervised_objectives=tuple(raw_before.keys()),
            raw_losses=raw_before,
            loss_before=loss_before,
            loss_after=loss_after,
            gradient_norm=grad_norm,
            changed_parameter_tensors=changed_tensors,
            parameter_update_norm=update_norm,
            model_revision_before=rev_before,
            model_revision_after=self.model_revision,
            reference_loss_before=ref_before,
            reference_loss_after=ref_after,
        )
        self.metrics_history.append(dataclasses.asdict(report))
        return report

    def learn_terminal_returns(
        self,
        episode_samples: Sequence[LearningSample],
        terminal_outcome: str | None,
    ) -> UpdateReport | None:
        """Applies discounted return supervision for completed episodes."""
        targets = return_targets_for_episode(
            len(episode_samples), terminal_outcome
        )
        if targets is None:
            return None

        rev_before = self.model_revision
        param_snap, opt_snap = self._snapshot_state()
        self.world.eval()
        with torch.no_grad():
            before_losses = self._episode_return_loss(episode_samples, targets)
        loss_before = float(before_losses.item())

        try:
            self.world.train()
            self.optimizer.zero_grad(set_to_none=True)
            train_loss = self._episode_return_loss(episode_samples, targets)
            if not torch.isfinite(train_loss).item():
                raise FloatingPointError('Non-finite return loss.')
            train_loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_params,
                    self.config.gradient_clip,
                    error_if_nonfinite=True,
                ).item()
            )
            self.optimizer.step()
            self.world.eval()
        except (FloatingPointError, RuntimeError) as exc:
            self._restore_snapshot(param_snap, opt_snap)
            self.world.eval()
            return UpdateReport(
                learning_sample_id=f'return-{episode_samples[-1].sample_id}',
                status='weight_update_rejected',
                supervised_objectives=('return',),
                raw_losses={'return': loss_before},
                loss_before=loss_before,
                loss_after=loss_before,
                gradient_norm=0.0,
                changed_parameter_tensors=0,
                parameter_update_norm=0.0,
                model_revision_before=rev_before,
                model_revision_after=rev_before,
                reference_loss_before=None,
                reference_loss_after=None,
                rejection_reason=str(exc),
            )

        changed_tensors = 0
        sq_diff_sum = 0.0
        for name, param in self.world.named_parameters():
            if name in param_snap:
                diff = param.detach() - param_snap[name]
                norm_sq = float((diff * diff).sum().item())
                if norm_sq > 0.0:
                    changed_tensors += 1
                    sq_diff_sum += norm_sq
        with torch.no_grad():
            loss_after = float(
                self._episode_return_loss(episode_samples, targets).item()
            )
        self.model_revision += 1
        self.supervision_counts['return'] += len(episode_samples)
        report = UpdateReport(
            learning_sample_id=f'return-{episode_samples[-1].sample_id}',
            status='weight_update_accepted',
            supervised_objectives=('return',),
            raw_losses={'return': loss_before},
            loss_before=loss_before,
            loss_after=loss_after,
            gradient_norm=grad_norm,
            changed_parameter_tensors=changed_tensors,
            parameter_update_norm=math.sqrt(sq_diff_sum),
            model_revision_before=rev_before,
            model_revision_after=self.model_revision,
            reference_loss_before=None,
            reference_loss_after=None,
        )
        self.metrics_history.append(dataclasses.asdict(report))
        return report

    def _episode_return_loss(
        self,
        episode_samples: Sequence[LearningSample],
        targets: tuple[float, ...],
    ) -> torch.Tensor:
        """Computes weighted discounted-return BCE across an episode."""
        step_losses: list[torch.Tensor] = []
        for sample, ret_val in zip(episode_samples, targets):
            eval_aligned = sample.after_aligned or sample.before_aligned
            belief = observation.observe(
                self.world,
                eval_aligned,
                previous=_detach_belief(sample.previous_belief),
            )
            belief = _with_context(belief, sample.context)
            goal_logits, _, _ = self.world.evaluate_goal(
                belief, sample.instruction_tokens, sample.instruction_mask
            )
            ret_logits = goal_logits[:, :, 1]
            ret_target = torch.full_like(ret_logits, ret_val)
            step_losses.append(
                functional.binary_cross_entropy_with_logits(
                    ret_logits, ret_target
                )
            )
        return LIVE_LOSS_WEIGHTS['return'] * torch.stack(step_losses).mean()

    def save_checkpoint(self, out_dir: pathlib.Path) -> pathlib.Path:
        """Atomically saves live checkpoint, config, metrics, and manifest."""
        out_dir.mkdir(parents=True, exist_ok=True)
        role_vocab = [
            role
            for role, _ in sorted(
                self.role_to_index.items(), key=lambda item: item[1]
            )
        ]
        param_count = sum(p.numel() for p in self.world.parameters())
        payload = {
            'base_checkpoint_sha256': self.base_checkpoint_sha256,
            'param_count': param_count,
            'role_vocabulary': role_vocab,
            'slot_count': self.world.slot_count,
            'model_revision': self.model_revision,
            'model_state_dict': self.world.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'supervision_counts': dict(self.supervision_counts),
            'trained_capabilities': self.capability_status(),
            'frozen_parameter_groups': list(self.frozen_parameter_groups),
            'updated_parameter_groups': list(self.updated_parameter_groups),
        }
        latest_path = out_dir / 'latest.pt'
        tmp_path = out_dir / 'latest.pt.tmp'
        torch.save(payload, tmp_path)
        os.replace(tmp_path, latest_path)

        (out_dir / 'config.json').write_text(
            json.dumps(dataclasses.asdict(self.config), indent=2),
            encoding='utf-8',
        )
        metrics_lines = '\n'.join(
            json.dumps(row) for row in self.metrics_history
        )
        if metrics_lines:
            metrics_lines += '\n'
        (out_dir / 'learning_metrics.jsonl').write_text(
            metrics_lines, encoding='utf-8'
        )
        manifest = {
            'base_checkpoint_sha256': self.base_checkpoint_sha256,
            'param_count': param_count,
            'model_revision': self.model_revision,
            'supervision_counts': dict(self.supervision_counts),
            'trained_capabilities': self.capability_status(),
            'frozen_parameter_count': len(self.frozen_parameter_groups),
            'updated_parameter_count': len(self.updated_parameter_groups),
        }
        (out_dir / 'manifest.json').write_text(
            json.dumps(manifest, indent=2), encoding='utf-8'
        )
        return latest_path


def load_from_live_checkpoint(
    ckpt_path: pathlib.Path,
    device: torch.device,
) -> tuple[model.WorldActionModel, dict[str, Any]]:
    """Strictly reloads a saved live-learning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    world = model.WorldActionModel(
        role_count=len(ckpt['role_vocabulary']),
        slot_count=int(ckpt['slot_count']),
    ).to(device)
    world.load_state_dict(ckpt['model_state_dict'], strict=True)
    return world, ckpt
