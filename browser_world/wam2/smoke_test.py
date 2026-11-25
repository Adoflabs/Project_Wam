"""Synthetic tensor smoke test, not evidence of learned intelligence."""

import torch
from torch.nn import functional

from browser_world.wam2 import model


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(2)
    batch, slots, roles = 2, 12, 16
    net = model.WorldActionModel(roles, slots)
    present = torch.zeros(batch, slots)
    present[:, :8] = 1
    parent = functional.one_hot(torch.zeros(batch, slots, dtype=torch.long), slots + 1).float()
    observed = dict(names=torch.randn(batch, slots, 384), values=torch.randn(batch, slots, 384),
        roles=functional.one_hot(torch.randint(roles, (batch, slots)), roles).float(),
        states=torch.randint(2, (batch, slots, 8)).float(), flags=torch.zeros(batch, slots, 4),
        boxes=torch.rand(batch, slots, 4), affordances=torch.ones(batch, slots, 4),
        parents=parent, present=present)
    belief = net.observe(**observed)
    kind = torch.tensor([2, 0]); target = torch.tensor([3, 4])
    text = torch.zeros(batch, 384); checked = torch.tensor([[1., 0., 0.]]).expand(batch, -1)
    scroll = torch.zeros(batch, 1); pointer = torch.zeros(batch, 2)
    tokens = torch.randn(batch, 7, 384); mask = torch.ones(batch, 7, dtype=torch.bool)
    original_belief = belief
    initial = belief.latent.detach().clone()
    losses = []
    for _ in range(3):
        action = net.encode_action(belief, kind, target, text, checked, scroll, pointer)
        prediction = net.imagine(belief, action, target)
        belief = prediction.belief
        goal, proposal, copy = net.evaluate_goal(belief, tokens, mask)
        assert goal.shape == (batch, 3, 2)
        assert proposal.shape == (batch, slots, 4)
        assert copy.shape == (batch, slots, 2, 7)
        assert belief.latent.shape == (batch, slots, 256)
        assert belief.parents.shape == (batch, slots, slots + 1)
        assert prediction.content_source_logits.shape == (
            batch, slots, 2, 5 + 2 * slots
        )
        assert torch.isfinite(belief.latent).all()
        assert ((belief.states >= 0) & (belief.states <= 1)).all()
        losses.append(goal.square().mean() + belief.states.mean() + belief.present.mean()
                      + belief.latent.square().mean())
    actual = net.observe(**observed, previous=original_belief, matched=present.bool())
    corrected = net.correct_context(actual, belief, action, present.bool())
    assert torch.count_nonzero(corrected.context) > 0
    assert torch.count_nonzero(belief.context) == 0
    inverse_kind, inverse_target = net.infer_action(original_belief, actual)
    assert inverse_kind.shape == (batch, 5)
    assert inverse_target.shape == (batch, slots + 1)
    loss = (sum(losses) + corrected.context.square().mean()
            + inverse_kind.square().mean() + inverse_target[:, :9].square().mean())
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    assert torch.equal(initial, original_belief.latent.detach())
    print('Three recursive predictions, goal/copy shapes, real-only context correction, finite backward: PASS')
    print('Parameters (16 roles, 128 slots):', sum(p.numel() for p in model.WorldActionModel(16).parameters()))
    print('Torch:', torch.__version__)


if __name__ == '__main__':
    main()
