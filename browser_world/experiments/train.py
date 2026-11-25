"""Full training loop and checkpointing for the one-step WorldModel."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import time
from typing import Any

import torch
from torch.utils import data as torch_data

from browser_world.dataset import schema
from browser_world.dataset import training_data
from browser_world.experiments import baselines
from browser_world.experiments import evaluate
from browser_world.model import losses
from browser_world.model import world_model

SEED = 0
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WINDOW_LENGTH = 1


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_dataset(
    dataset_dir: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest_path = dataset_dir / 'manifest.json'
    vocab_path = dataset_dir / 'vocab.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    vocab = json.loads(vocab_path.read_text(encoding='utf-8'))

    if manifest.get('input_encoding') != 'utf8_bytes_v1':
        raise ValueError(
            f"Unsupported input_encoding: {manifest.get('input_encoding')!r}"
        )
    cfg = manifest['config']
    if (
        tuple(cfg['state_names']) != schema.STATE_NAMES
        or tuple(cfg['action_kinds']) != schema.ACTION_KINDS
        or tuple(cfg['value_sources']) != schema.VALUE_SOURCES
        or tuple(cfg['outcomes']) != schema.OUTCOMES
    ):
        raise ValueError('Dataset manifest schema does not match schema.py.')

    vocab_sizes = manifest['vocab_sizes']
    if (
        len(vocab['roles']) != vocab_sizes['roles']
        or len(vocab['words']) != vocab_sizes['words']
        or len(vocab['chars']) != vocab_sizes['chars']
    ):
        raise ValueError('Vocabulary sizes do not match manifest.')
    if _sha256(vocab_path) != manifest['vocab_sha256']:
        raise ValueError('vocab.json hash mismatch.')
    for split_file in ('train.jsonl', 'validation.jsonl'):
        if _sha256(dataset_dir / split_file) != (
            manifest['curated_files'][split_file]
        ):
            raise ValueError(f'{split_file} hash mismatch.')

    return manifest, vocab, _sha256(manifest_path)


def train_epoch(
    model: world_model.WorldModel,
    loader: torch_data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    action_blind: bool = False,
) -> dict[str, float]:
    """Updates model parameters for one complete training epoch."""
    model.train()
    totals: dict[str, float] = {}
    batch_count = 0

    for cpu_batch in loader:
        batch = {
            key: value.to(device)
            for key, value in cpu_batch.items()
        }
        inputs = evaluate.model_inputs(batch, action_blind=action_blind)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(**inputs)
        total_loss, terms = losses.prediction_loss(predictions, batch)
        if not torch.isfinite(total_loss):
            raise RuntimeError('Nonfinite training loss.')
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=MAX_GRAD_NORM,
            error_if_nonfinite=True,
        )
        optimizer.step()

        measurements = {
            'total': total_loss.detach().item(),
            'gradient_norm': grad_norm.item(),
        }
        measurements.update({
            name: value.detach().item()
            for name, value in terms.items()
        })
        for name, value in measurements.items():
            totals[name] = totals.get(name, 0.0) + value
        batch_count += 1

    if batch_count == 0:
        raise ValueError('Training loader is empty.')

    return {
        name: value / batch_count
        for name, value in totals.items()
    }


def _baseline_validation(
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    results = {}
    for name, fn in (
        ('no_change', baselines.predict_no_change),
        ('action_copy', baselines.predict_action_copy),
    ):
        base = baselines.evaluate(examples, fn)
        nt_tp = nt_fp = nt_fn = 0
        for ex in examples:
            pred = fn(ex)
            target_idx = int(ex['inputs']['action']['target'])
            for idx, tgt in enumerate(ex['targets']['elements']):
                if not tgt['known'] or idx == target_idx:
                    continue
                t_c, p_c = bool(tgt['changed']), bool(pred['changed'][idx])
                nt_tp += int(t_c and p_c)
                nt_fp += int(p_c and not t_c)
                nt_fn += int(t_c and not p_c)
        denom = 2 * nt_tp + nt_fp + nt_fn
        results[name] = {
            'baseline_format': base,
            'non_target_changed': {
                'precision': nt_tp / (nt_tp + nt_fp) if (nt_tp + nt_fp) else 0.0,
                'recall': nt_tp / (nt_tp + nt_fn) if (nt_tp + nt_fn) else 0.0,
                'f1': 2 * nt_tp / denom if denom else 0.0,
                'tp': nt_tp,
                'fp': nt_fp,
                'fn': nt_fn,
            },
        }
    return results


def _save_checkpoint(
    path: pathlib.Path,
    epoch: int,
    num_roles: int,
    model: world_model.WorldModel,
    optimizer: torch.optim.Optimizer,
    role_vocabulary: list[str],
    manifest_hash: str,
    validation_metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            'format_version': 1,
            'epoch': epoch,
            'num_roles': num_roles,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'role_vocabulary': role_vocabulary,
            'input_keys': list(world_model.INPUT_KEYS),
            'dataset_manifest_sha256': manifest_hash,
            'validation_metrics': validation_metrics,
            'config': config,
        },
        path,
    )


def run_training(
    dataset_dir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    action_blind: bool = False,
) -> dict[str, Any]:
    """Runs the 50-epoch training and validation experiment on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for M0.1g full training.')
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest, vocab, manifest_hash = _validate_dataset(dataset_dir)
    num_roles = len(vocab['roles'])

    train_dataset = training_data.Windows.load(
        dataset_dir, 'train', WINDOW_LENGTH,
    )
    validation_dataset = training_data.Windows.load(
        dataset_dir, 'validation', WINDOW_LENGTH,
    )
    train_examples = training_data.load_examples(dataset_dir, 'train')
    validation_examples = training_data.load_examples(dataset_dir, 'validation')

    if len(train_dataset) != 343 or len(validation_dataset) != 82:
        raise ValueError(
            f'Unexpected split sizes: train={len(train_dataset)}, '
            f'validation={len(validation_dataset)}'
        )
    train_runs = {e['meta']['run_id'] for e in train_examples}
    val_runs = {e['meta']['run_id'] for e in validation_examples}
    if train_runs & val_runs:
        raise ValueError('Train and validation run_ids overlap.')

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = torch_data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
        collate_fn=training_data.collate,
    )
    validation_loader = torch_data.DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=training_data.collate,
    )

    device = torch.device('cuda:0')
    model = world_model.WorldModel(num_roles=num_roles).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    if param_count != 675802:
        raise ValueError(f'Unexpected parameter count: {param_count}')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config: dict[str, Any] = {
        'model': 'WorldModel',
        'dataset_dir': dataset_dir.as_posix(),
        'num_roles': num_roles,
        'parameter_count': param_count,
        'seed': SEED,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'max_grad_norm': MAX_GRAD_NORM,
        'window_length': WINDOW_LENGTH,
        'action_blind': action_blind,
        'precision': 'fp32',
        'device': torch.cuda.get_device_name(device),
        'torch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda,
        'git_commit': _git_commit(),
        'git_dirty': _git_dirty(),
        'dataset_manifest_sha256': manifest_hash,
        'vocab_sha256': manifest['vocab_sha256'],
        'train_jsonl_sha256': manifest['curated_files']['train.jsonl'],
        'validation_jsonl_sha256': manifest['curated_files']['validation.jsonl'],
    }
    with (out_dir / 'config.json').open(
        'w', encoding='utf-8', newline='\n'
    ) as f:
        f.write(json.dumps(config, indent=2))

    baseline_results = _baseline_validation(validation_examples)

    history_path = out_dir / 'history.jsonl'
    best_path = out_dir / 'best.pt'
    last_path = out_dir / 'last.pt'

    epoch0_val = evaluate.evaluate_model(
        model, validation_loader, validation_dataset,
        validation_examples, device, vocab['roles'],
        action_blind=action_blind,
    )
    best_f1 = float(epoch0_val['non_target_changed']['f1'])
    best_epoch = 0
    best_metrics = epoch0_val
    _save_checkpoint(
        best_path, 0, num_roles, model, optimizer,
        vocab['roles'], manifest_hash, epoch0_val, config,
    )

    with history_path.open('w', encoding='utf-8', newline='\n') as hist_f:
        hist_f.write(json.dumps({
            'epoch': 0,
            'train': None,
            'validation': epoch0_val,
        }) + '\n')

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    updates = 0
    last_val = epoch0_val
    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            action_blind=action_blind,
        )
        updates += len(train_loader)
        val_metrics = evaluate.evaluate_model(
            model, validation_loader, validation_dataset,
            validation_examples, device, vocab['roles'],
            action_blind=action_blind,
        )
        last_val = val_metrics

        with history_path.open('a', encoding='utf-8', newline='\n') as hist_f:
            hist_f.write(json.dumps({
                'epoch': epoch,
                'train': train_metrics,
                'validation': val_metrics,
            }) + '\n')

        val_nt_f1 = float(val_metrics['non_target_changed']['f1'])
        if val_nt_f1 > best_f1:
            best_f1 = val_nt_f1
            best_epoch = epoch
            best_metrics = val_metrics
            _save_checkpoint(
                best_path, epoch, num_roles, model, optimizer,
                vocab['roles'], manifest_hash, val_metrics, config,
            )

        print(
            f'Epoch {epoch:2d} | train={train_metrics["total"]:.4f} '
            f'| val={val_metrics["loss"]["total"]:.4f} '
            f'| chg_f1={val_metrics["baseline_format"]["changed_element"]["f1"]:.3f} '
            f'| nt_f1={val_nt_f1:.3f} '
            f'| rem_f1={val_metrics["baseline_format"]["removed_element"]["f1"]:.3f} '
            f'| new_mae={val_metrics["baseline_format"]["new_count_mae"]:.3f}'
        )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_alloc_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_res_mib = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    _save_checkpoint(
        last_path, EPOCHS, num_roles, model, optimizer,
        vocab['roles'], manifest_hash, last_val, config,
    )

    summary = {
        'elapsed_seconds': elapsed,
        'optimizer_updates': updates,
        'peak_cuda_allocated_mib': peak_alloc_mib,
        'peak_cuda_reserved_mib': peak_res_mib,
        'gpu_device': torch.cuda.get_device_name(device),
        'best_epoch': best_epoch,
        'baselines': baseline_results,
        'epoch_0': epoch0_val,
        'best': best_metrics,
        'last': last_val,
    }
    with (out_dir / 'metrics.json').open(
        'w', encoding='utf-8', newline='\n'
    ) as f:
        f.write(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """CLI entry point for M0.1g training."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='data/curated/bytes_v1')
    parser.add_argument('--out', default='artifacts/bw0/m01g_seed0')
    parser.add_argument('--action-blind', action='store_true')
    args = parser.parse_args()
    run_training(
        pathlib.Path(args.dataset),
        pathlib.Path(args.out),
        action_blind=args.action_blind,
    )


if __name__ == '__main__':
    main()
