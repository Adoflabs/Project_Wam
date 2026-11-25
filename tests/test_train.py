"""Unit tests for the training loop and checkpoint checks."""

# pylint: disable=missing-function-docstring

import pytest
import torch
from torch.utils import data as torch_data

from browser_world.dataset import training_data
from browser_world.experiments import evaluate
from browser_world.experiments import train
from browser_world.model import world_model
from tests import test_evaluate


def test_train_epoch_updates_params_and_val_preserves_them():
    ex_a = test_evaluate._synthetic_example('r1', 0, num_elems=3, target=0)  # pylint: disable=protected-access
    ex_b = test_evaluate._synthetic_example('r2', 0, num_elems=2, target=1)  # pylint: disable=protected-access
    dataset = training_data.Windows([[ex_a], [ex_b]])
    loader = torch_data.DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=training_data.collate,
    )
    torch.manual_seed(0)
    model = world_model.WorldModel(num_roles=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    before = [p.detach().clone() for p in model.parameters()]
    train_metrics = train.train_epoch(
        model, loader, optimizer, torch.device('cpu'),
    )
    assert train_metrics['total'] > 0.0
    assert train_metrics['gradient_norm'] > 0.0
    assert any(
        not torch.equal(b, p.detach())
        for b, p in zip(before, model.parameters())
    )

    after_train = [p.detach().clone() for p in model.parameters()]
    evaluate.evaluate_model(
        model, loader, dataset, [ex_a, ex_b], torch.device('cpu'),
    )
    for b, p in zip(after_train, model.parameters()):
        assert torch.equal(b, p.detach())


def test_nonfinite_loss_rejected_and_checkpoint_roundtrip(tmp_path):
    ex = test_evaluate._synthetic_example('r1', 0, num_elems=2, target=0)  # pylint: disable=protected-access
    dataset = training_data.Windows([[ex]])
    loader = torch_data.DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=training_data.collate,
    )
    torch.manual_seed(0)
    model = world_model.WorldModel(num_roles=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    ckpt_path = tmp_path / 'best.pt'
    torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
    reloaded = world_model.WorldModel(num_roles=16)
    state = torch.load(ckpt_path, weights_only=True, map_location='cpu')
    reloaded.load_state_dict(state['model_state_dict'])
    m_orig = evaluate.evaluate_model(
        model, loader, dataset, [ex], torch.device('cpu'),
    )
    m_reload = evaluate.evaluate_model(
        reloaded, loader, dataset, [ex], torch.device('cpu'),
    )
    assert m_orig['loss']['total'] == pytest.approx(m_reload['loss']['total'])

    # Inject NaN into a parameter to verify nonfinite loss guard.
    with torch.no_grad():
        model.heads.changed.weight.fill_(float('nan'))
    with pytest.raises(RuntimeError, match='Nonfinite'):
        train.train_epoch(model, loader, optimizer, torch.device('cpu'))


def test_nonempty_output_directory_rejected(tmp_path):
    nonempty = tmp_path / 'existing'
    nonempty.mkdir()
    (nonempty / 'file.txt').write_text('keep', encoding='utf-8')
    if torch.cuda.is_available():
        with pytest.raises(FileExistsError):
            train.run_training(tmp_path / 'unused', nonempty)


def test_checkpoint_metadata_portability(tmp_path):
    torch.manual_seed(0)
    model = world_model.WorldModel(num_roles=16)
    ckpt_path = tmp_path / 'inference.pt'
    torch.save(
        {
            'format_version': 1,
            'epoch': 0,
            'num_roles': 16,
            'model_state_dict': model.state_dict(),
            'torch_version': str(torch.__version__),
        },
        ckpt_path,
    )
    loaded = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    assert loaded['torch_version'] == str(torch.__version__)
    reloaded = world_model.WorldModel(num_roles=loaded['num_roles'])
    reloaded.load_state_dict(loaded['model_state_dict'], strict=True)
