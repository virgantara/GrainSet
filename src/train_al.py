#!/usr/bin/env python3
"""GrainSet Rice: sequential pool-based Active Learning or a single-label-budget run.

Single mode trains one target budget directly. Uncertainty methods first require
one shared small labeled-set warmup model; Random needs only one training.
ResNet50 from original GrainSet, supervised ImageNet initial weights, no BYOL.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from al.acquisition import METHODS, select_indices
from al.data import (LabeledDataset, UnlabeledDataset, budgets, complement,
                     make_transforms, read_manifest, verify_manifests)
from al.engine import (classify, inference, load_weights, make_loader,
                       make_model, seed_everything, train_round)
from al.purity import bag_metrics

ROOT = Path(__file__).resolve().parents[1]


def args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-root', default=str(ROOT/'datasets/rice'), help='Real image root; e.g. /datasets/rice')
    p.add_argument('--output', default=str(ROOT/'runs/active_learning'))
    p.add_argument('--methods', nargs='+', choices=METHODS, default=None,
                   help='AL methods; in --single-pct mode exactly ONE method is allowed.')
    p.add_argument('--method', choices=METHODS, default=None,
                   help='Single-method shortcut, e.g. --method entropy.')
    p.add_argument('--seeds', nargs='+', type=int, default=None,
                   help='Random seeds. In --single-pct mode exactly ONE seed is allowed.')
    p.add_argument('--seed', type=int, default=None,
                   help='Single-seed shortcut, e.g. --seed 42.')
    p.add_argument('--single-pct', type=int, default=None,
                   help='Train ONLY this labeled budget, e.g. --single-pct 5; no sequential AL rounds.')
    p.add_argument('--continue-from-pct', type=int, default=None,
                   help='Progressive single-budget AL: reuse previous labeled set and checkpoint, e.g. 5 then 10. One classifier training per invocation.')
    p.add_argument('--single-initial-pct', type=int, default=2,
                   help='Initial labeled budget for uncertainty/hybrid queries in single mode. A shared warmup model is trained once per seed.')
    p.add_argument('--single-eval-test', action='store_true',
                   help='Also evaluate the fixed test set for a single-budget run (do not use test results for model selection).')
    p.add_argument('--start-pct', type=int, default=2)
    p.add_argument('--max-pct', type=int, default=50)
    p.add_argument('--step-pct', type=int, default=2)
    p.add_argument('--max-rounds', type=int, default=None, help='Optional quick run; preserves full budget schedule for later resume')
    p.add_argument('--include-full', action='store_true', help='Full supervised 100%% reference (once per seed)')
    p.add_argument('--backend', choices=['timm','torchvision'], default='timm',
                   help='timm uses original repo src/models/model.py; torchvision is a debug fallback')
    p.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True,
                   help='ImageNet supervised initialization, NO BYOL')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--pool-batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=0,
                   help='DataLoader worker processes; 0 (default) prevents Too many open files.')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--candidate-multiplier', type=int, default=3)
    p.add_argument('--keep-checkpoints', choices=['latest','all'], default='latest')
    p.add_argument('--device', default='auto', help='auto, cuda, cuda:0 or cpu')
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Check manifests and print budgets; do not train')
    p.add_argument('--skip-file-check', action='store_true', help='Only for manifest/planning audit')
    p.add_argument('--bag-repetitions', type=int, default=30)
    p.add_argument('--bag-sizes', type=int, nargs='+', default=[100, 200, 500])
    p.add_argument('--bag-rates', type=float, nargs='+', default=[0.01, 0.02, 0.05, 0.10])
    args = p.parse_args()
    if args.method is not None:
        if args.methods is not None:
            p.error('Use --method OR --methods, not both')
        args.methods = [args.method]
    if args.seed is not None:
        if args.seeds is not None:
            p.error('Use --seed OR --seeds, not both')
        args.seeds = [args.seed]
    if args.methods is None:
        args.methods = ['random'] if args.single_pct is not None else list(METHODS)
    if args.seeds is None:
        args.seeds = [42] if args.single_pct is not None else [42, 123, 509]
    if args.single_pct is not None:
        if len(args.methods) != 1:
            p.error('Single run accepts ONE method. Use --method entropy (NOT --methods random entropy hybrid).')
        if len(args.seeds) != 1:
            p.error('Single run accepts ONE seed. Use --seed 42.')
    if args.continue_from_pct is not None and args.single_pct is None:
        p.error('--continue-from-pct is only valid with --single-pct')
    return args


def atomic_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w', encoding='utf8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def config_key(args, data_root):
    fields = dict(data_root=str(Path(data_root).resolve()), backend=args.backend,
                  pretrained=args.pretrained, epochs=args.epochs, patience=args.patience,
                  batch_size=args.batch_size, pool_batch_size=args.pool_batch_size,
                  lr=args.lr, weight_decay=args.weight_decay, image_size=args.image_size,
                  candidate_multiplier=args.candidate_multiplier, deterministic=args.deterministic,
                  start_pct=args.start_pct, max_pct=args.max_pct, step_pct=args.step_pct,
                  bag_repetitions=args.bag_repetitions, bag_sizes=args.bag_sizes, bag_rates=args.bag_rates)
    return fields


def evaluate_and_save(model, manifest, eval_transform, args, device, output_dir, prefix, seed):
    ds = LabeledDataset(manifest, np.arange(len(manifest)), eval_transform)
    loader = make_loader(ds, args.pool_batch_size, args.workers, device=device)
    idx, y, prob, _ = inference(model, loader, device)
    if not np.array_equal(idx, np.arange(len(manifest))):
        raise AssertionError('Evaluation order changed unexpectedly')
    metrics, cm = classify(y, prob)
    purity = bag_metrics(y, prob, bag_sizes=args.bag_sizes, rates=args.bag_rates,
                         repetitions=args.bag_repetitions, seed=2026)
    metrics.update(purity)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/f'{prefix}_predictions.npz', paths=np.array(manifest.paths),
                        y_true=y, probabilities=prob, y_pred=prob.argmax(axis=1))
    np.savetxt(out/f'{prefix}_confusion.csv', cm, fmt='%d', delimiter=',')
    atomic_json(metrics, out/f'{prefix}_metrics.json')
    return metrics


def acquire(model, manifest, labeled, n_required, method, args, eval_tf, device, seed, round_i):
    pool = complement(len(manifest), labeled)
    if method == 'random':
        return select_indices('random', pool, n_required, None, None, seed+10007*round_i)
    unlabeled = UnlabeledDataset(manifest.paths, pool, eval_tf)
    loader = make_loader(unlabeled, args.pool_batch_size, args.workers, device=device)
    observed, labels, probabilities, embeddings = inference(
        model, loader, device, want_features=(method == 'hybrid'))
    if len(labels):
        raise AssertionError('Unlabeled inference must not expose ground-truth labels')
    if not np.array_equal(observed, pool):
        raise AssertionError('Unexpected pool inference order')
    return select_indices(method, pool, n_required, probabilities, embeddings,
                          seed=seed+10007*round_i,
                          candidate_multiplier=args.candidate_multiplier)



def single_settings(args):
    """Settings saved per run; prevents silently resuming with changed hyperparams."""
    return {**config_key(args, args.data_root), 'single_pct': args.single_pct,
            'single_initial_pct': args.single_initial_pct,
            'continue_from_pct': args.continue_from_pct,
            'single_eval_test': args.single_eval_test}


def single_config_matches(saved, current):
    """Backward compatible with single-pct runs saved before progressive mode."""
    old = dict(saved)
    new = dict(current)
    old.setdefault('continue_from_pct', None)
    new.setdefault('continue_from_pct', None)
    return old == new


def can_reuse_legacy_random_target(saved_config, current_config, method, seed,
                                   initial_indices, target_indices, previous_pct,
                                   output_root, train_size):
    """Accept completed legacy Random runs without changing any saved files.

    Legacy single-budget Random sampling used a prefix of a fixed permutation,
    so it is nested: the 2% selection is included in the already trained 5%.
    Only `continue_from_pct` may differ; all model/data hyperparameters must
    match and the requested previous labeled set must be a subset of the target.
    """
    if method != 'random' or previous_pct is None:
        return False, 'not a legacy Random continuation'
    old = dict(saved_config)
    old.setdefault('continue_from_pct', None)
    if old.get('continue_from_pct') is not None:
        return False, 'legacy run was itself progressive from a different percentage'
    current_without_parent = dict(current_config)
    current_without_parent['continue_from_pct'] = None
    if old != current_without_parent:
        return False, 'training settings differ beyond continue-from-pct'

    parent = output_root.parent / f'single_{previous_pct}pct' / method / f'seed_{seed}'
    parent_result, parent_selection = parent / 'result.json', parent / 'selection.json'
    if not parent_result.is_file() or not parent_selection.is_file():
        return False, f'previous {previous_pct}% Random result/selection not found: {parent}'
    parent_info = json.loads(parent_result.read_text(encoding='utf8'))
    parent_pool = json.loads(parent_selection.read_text(encoding='utf8'))
    if parent_pool.get('method') != method or int(parent_pool.get('seed', -1)) != int(seed):
        return False, 'previous method or seed differs'
    if any(parent_info.get('config', {}).get(k) != v
           for k, v in config_key_from_single(current_config).items()):
        return False, 'previous training settings differ'
    parent_indices = [int(i) for i in parent_pool['labeled_indices']]
    expected_parent = round(train_size * previous_pct / 100)
    if len(parent_indices) != expected_parent or len(set(parent_indices)) != expected_parent:
        return False, 'previous labeled pool size is wrong or contains duplicates'
    if not set(initial_indices).issubset(parent_indices):
        return False, 'previous pool does not contain initial labeled samples'
    if not set(parent_indices).issubset(target_indices):
        return False, 'existing target pool does not contain previous labeled pool'
    return True, 'validated nested Random subsets and unchanged model/data settings'


def config_key_from_single(config):
    """Base hyperparameters, excluding metadata specific to the single run."""
    keys = {'single_pct', 'single_initial_pct', 'continue_from_pct', 'single_eval_test'}
    return {k: v for k, v in config.items() if k not in keys}


def single_init_indices(train_size, seed, count):
    # Same initial samples for random and all uncertainty methods at a given seed.
    return np.random.default_rng(seed).permutation(train_size)[:count].tolist()


def save_single_sample_list(file_path, manifest, indices):
    csv_write(file_path, [{'index': int(i), 'path': manifest.paths[i],
                           'revealed_label': int(manifest.labels[i])}
                          for i in indices])


def run_single_warmup(seed, train, val, args, device, train_tf, eval_tf,
                      initial_indices, output_root):
    """Train one shared supervised seed model for uncertainty acquisition.

    A 5% entropy/hybrid experiment cannot acquire 5% labels from an untrained
    classifier; initial 2% supervised warmup is a real prerequisite.
    """
    warm_dir = output_root/'_warmup'/f'seed_{seed}'
    warm_dir.mkdir(parents=True, exist_ok=True)
    result_file = warm_dir/'result.json'
    checkpoint = warm_dir/'best.pt'
    settings = single_settings(args)
    if result_file.exists():
        saved = json.loads(result_file.read_text(encoding='utf8'))
        if not single_config_matches(saved['config'], settings) or saved['initial_indices'] != initial_indices:
            raise RuntimeError(f'Changed warmup settings at {result_file}. Use another --output directory.')
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Warmup checkpoint not found: {checkpoint}')
        print(f'[warmup seed={seed}] Reusing trained {len(initial_indices)}-image model: {checkpoint}')
        return checkpoint
    print(f'\n[warmup seed={seed}] Train ResNet50 on {len(initial_indices)} initial labels', flush=True)
    round_seed = int(seed) + 300003
    seed_everything(round_seed, args.deterministic)
    model = make_model(args.backend, pretrained=args.pretrained and not (warm_dir/'last.pt').exists()).to(device)
    info = train_round(model,
                       LabeledDataset(train, initial_indices, train_tf),
                       LabeledDataset(val, np.arange(len(val)), eval_tf),
                       args, warm_dir, device, round_seed)
    save_single_sample_list(warm_dir/'initial_samples.csv', train, initial_indices)
    atomic_json({'config': settings, 'seed': seed, 'initial_indices': initial_indices,
                 'best_epoch': info['best_epoch']}, result_file)
    del model
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    return checkpoint


def run_single(method, seed, train, val, test, args, device, train_tf, eval_tf, output_root):
    """Train exactly one TARGET labeled-budget classifier, no prior AL rounds.

    Random:  one classifier training at the requested percentage.
    Others: a shared supervised initial-pct model -> query to target -> one
            target classifier training, for a scientifically valid AL query.
    """
    target = round(len(train) * args.single_pct / 100)
    start_count = round(len(train) * args.single_initial_pct / 100)
    if not (1 <= start_count <= target <= len(train)):
        raise ValueError('Initial labeled count must be >=1 and <= target')
    initial = single_init_indices(len(train), seed, start_count)
    settings = single_settings(args)
    out = output_root/method/f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    result_file = out/'result.json'
    selection_file = out/'selection.json'
    if result_file.exists():
        saved = json.loads(result_file.read_text(encoding='utf8'))
        if not single_config_matches(saved['config'], settings):
            # A previously completed random 5% run may have been produced
            # before the explicit progressive option was added. As long as
            # the 2% labeled set is a subset and the model/data settings are
            # identical, it remains a valid Random baseline: do NOT retrain.
            reusable = False
            reason = 'could not inspect the saved sample selection'
            if selection_file.is_file():
                old_selection = json.loads(selection_file.read_text(encoding='utf8'))
                target_ids = [int(i) for i in old_selection.get('labeled_indices', [])]
                if (old_selection.get('method') == method and
                        int(old_selection.get('seed', -1)) == int(seed) and
                        len(target_ids) == target and len(set(target_ids)) == target and
                        single_config_matches(old_selection.get('config', {}), saved['config'])):
                    reusable, reason = can_reuse_legacy_random_target(
                        saved['config'], settings, method, seed, initial,
                        target_ids, args.continue_from_pct, output_root, len(train))
                else:
                    reason = 'existing target selection differs from completed run'
            if reusable:
                print(f'[single {method} seed={seed}] Existing {args.single_pct}% '
                      f'Random model is compatible with progressive '
                      f'{args.continue_from_pct}% -> {args.single_pct}% ({reason}).')
                print('  Reusing completed training unchanged; no checkpoints or results overwritten.')
                return
            raise RuntimeError(
                f'Changed settings at {result_file}. Cannot safely reuse existing '
                f'{args.single_pct}% result ({reason}). Existing files are untouched. '
                f'Use a different --output directory to retrain.')
        print(f'[single {method} seed={seed}] {args.single_pct}% already completed; skipping')
        return

    if selection_file.exists():
        selection = json.loads(selection_file.read_text(encoding='utf8'))
        if not single_config_matches(selection['config'], settings) or selection['seed'] != seed or selection['method'] != method:
            raise RuntimeError(f'Saved selection has different settings: {selection_file}')
        labeled = [int(i) for i in selection['labeled_indices']]
        queried = [int(i) for i in selection['queried_indices']]
        prev_count = int(selection.get('previous_labeled_count', start_count))
        if len(labeled) != target or len(set(labeled)) != target:
            raise AssertionError('Saved target pool has incorrect count / duplicates')
        if not set(initial).issubset(labeled):
            raise AssertionError('Saved selected pool does not contain the initial 2% set')
        if len(queried) != target-prev_count:
            raise AssertionError('Saved acquired sample count is inconsistent')
        print(f'[single {method} seed={seed}] Reusing saved selection {args.single_pct}%')
    else:
        if args.continue_from_pct is not None:
            old_pct = int(args.continue_from_pct)
            if not (1 <= old_pct < args.single_pct):
                raise ValueError('--continue-from-pct must be lower than --single-pct')
            prev_dir = output_root.parent/f'single_{old_pct}pct'/method/f'seed_{seed}'
            prev_selection = prev_dir/'selection.json'
            prev_result = prev_dir/'result.json'
            prev_checkpoint = prev_dir/'best.pt'
            # Random acquisition needs the previous selected pool, not model weights.
            required = (prev_selection, prev_result) if method == 'random' else (prev_selection, prev_result, prev_checkpoint)
            missing = [str(f) for f in required if not f.is_file()]
            if missing:
                raise FileNotFoundError(
                    f'Cannot continue {old_pct}% -> {args.single_pct}%. Missing: {missing}. '
                    f'Run --single-pct {old_pct} --method {method} --seed {seed} first.')
            sel = json.loads(prev_selection.read_text(encoding='utf8'))
            result = json.loads(prev_result.read_text(encoding='utf8'))
            if sel.get('method') != method or int(sel.get('seed', -1)) != seed:
                raise RuntimeError('Previous selection does not match method or seed')
            # Prevent silently comparing AL runs with differing model/training settings.
            prev_cfg = result.get('config', {})
            current_cfg = config_key(args, args.data_root)
            if any(prev_cfg.get(k) != v for k, v in current_cfg.items()):
                raise RuntimeError('Training hyperparameters differ from previous percentage. '
                                   'Use the same parameters and output directory.')
            previous = [int(i) for i in sel['labeled_indices']]
            expected = round(len(train)*old_pct/100)
            if len(previous) != expected or len(set(previous)) != expected:
                raise AssertionError('Previous labeled set size or uniqueness is invalid')
            if not set(initial).issubset(previous):
                raise AssertionError('Previous labeled set differs from initial seed set')
            prev_count = len(previous)
            needed = target-prev_count
            if method == 'random':
                queried = acquire(None, train, previous, needed, method, args,
                                  eval_tf, device, seed, round_i=args.single_pct).tolist()
            else:
                query_model = make_model(args.backend, pretrained=False).to(device)
                load_weights(query_model, prev_checkpoint, device)
                queried = acquire(query_model, train, previous, needed, method, args,
                                  eval_tf, device, seed, round_i=args.single_pct).tolist()
                del query_model
                if str(device).startswith('cuda'): torch.cuda.empty_cache()
            labeled = previous + queried
            print(f'[single {method} seed={seed}] Incremental AL '
                  f'{old_pct}% -> {args.single_pct}%: +{needed} labels', flush=True)
        elif method == 'random':
            labeled = single_init_indices(len(train), seed, target)
            queried = labeled[start_count:]
            prev_count = start_count
        else:
            if target == start_count:
                labeled, queried = initial, []
            else:
                # Direct single-budget mode needs a 2% labeled warmup for
                # uncertainty sampling. For exactly one training per command,
                # first train 2%, then use --continue-from-pct 2.
                checkpoint = run_single_warmup(seed, train, val, args, device,
                                              train_tf, eval_tf, initial, output_root)
                query_model = make_model(args.backend, pretrained=False).to(device)
                load_weights(query_model, checkpoint, device)
                queried = acquire(query_model, train, initial, target-start_count,
                                  method, args, eval_tf, device, seed, round_i=1).tolist()
                del query_model
                if str(device).startswith('cuda'): torch.cuda.empty_cache()
                labeled = initial + queried
            prev_count = start_count
        if len(labeled) != target or len(set(labeled)) != target:
            raise AssertionError('Selected pool has incorrect count / duplicates')
        if any(i < 0 or i >= len(train) for i in labeled):
            raise AssertionError('Selected pool contains out-of-bounds index')
        atomic_json({'config': settings, 'method': method, 'seed': seed,
                     'initial_indices': initial, 'queried_indices': queried,
                     'previous_labeled_count': prev_count,
                     'continued_from_pct': args.continue_from_pct,
                     'labeled_indices': labeled}, selection_file)
    save_single_sample_list(out/'labeled_samples.csv', train, labeled)
    print(f'\n[single {method} seed={seed}] Target={args.single_pct}% '
          f'({target} labels); initial={start_count}, queried={len(queried)}', flush=True)
    round_seed = int(seed) + 300003
    seed_everything(round_seed, args.deterministic)
    model = make_model(args.backend, pretrained=args.pretrained and not (out/'last.pt').exists()).to(device)
    tic = time.perf_counter()
    info = train_round(model, LabeledDataset(train, labeled, train_tf),
                       LabeledDataset(val, np.arange(len(val)), eval_tf),
                       args, out, device, round_seed)
    val_info = evaluate_and_save(model, val, eval_tf, args, device, out, 'val', seed)
    row = {'method': method, 'seed': seed, 'round': 0,
           'label_pct': args.single_pct, 'labeled_count': target,
           'initial_count': start_count, 'acquired': len(queried),
           'training_seconds': round(time.perf_counter()-tic,2),
           'best_epoch': info['best_epoch'],
           **{f'val_{k}': v for k, v in val_info.items()}}
    if args.single_eval_test:
        test_info = evaluate_and_save(model, test, eval_tf, args, device, out, 'test', seed)
        row.update({f'test_{k}': v for k, v in test_info.items()})
    csv_write(out/'history.csv', [row])
    atomic_json({'config': settings, 'metrics': row}, result_file)
    print(f"  val top1={val_info['top1']:.2f} top2={val_info['top2']:.2f} "
          f"macro_f1={val_info['macro_f1']:.2f} MAPE={val_info['clean_mape_soft']:.3f}%", flush=True)
    del model
    if str(device).startswith('cuda'): torch.cuda.empty_cache()



def run_al(method, seed, train, val, test, args, device, train_tf, eval_tf, schedule, output_root):
    output_dir = output_root/method/f'seed_{seed}'
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir/'state.json'
    settings = config_key(args, args.data_root)
    initial = np.random.default_rng(seed).permutation(len(train))[:schedule[0][1]].tolist()
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding='utf8'))
        if state['config'] != settings:
            raise RuntimeError(f'Changed settings at {state_file}. Use another --output directory.')
        labeled = list(map(int, state['labeled_indices']))
        completed = int(state['completed_round'])
        history = state['history']
        if completed >= 0 and len(labeled) != schedule[completed][1]:
            raise AssertionError('Saved pool size differs from expected budget')
        if not set(initial).issubset(set(labeled)):
            raise AssertionError('Initial pool differs from stored state')
    else:
        labeled, completed, history = initial, -1, []
        atomic_json({'config': settings, 'method': method, 'seed': seed,
                     'initial_indices': initial, 'labeled_indices': labeled,
                     'completed_round': completed, 'history': history}, state_file)
    limit = len(schedule) if args.max_rounds is None else min(args.max_rounds, len(schedule))
    for i in range(completed+1, limit):
        pct, target = schedule[i]
        round_dir = output_dir/f'round_{i:02d}'
        round_dir.mkdir(parents=True, exist_ok=True)
        query_file = round_dir/'queried_indices.json'
        needed = target-len(labeled)
        if needed < 0: raise AssertionError('Invalid label budget/state')
        if query_file.exists():
            new_ids = [int(j) for j in json.loads(query_file.read_text())]
        elif i == 0:
            new_ids = []
            atomic_json(new_ids, query_file)
        else:
            # Earlier checkpoint is indispensable for uncertainty queries.
            previous = output_dir/f'round_{i-1:02d}'/'best.pt'
            if method != 'random' and not previous.exists():
                raise FileNotFoundError(f'Previous AL checkpoint missing: {previous}')
            if method == 'random':
                new_ids = acquire(None, train, labeled, needed, method, args,
                                  eval_tf, device, seed, i).tolist()
            else:
                previous_model = make_model(args.backend, pretrained=False).to(device)
                load_weights(previous_model, previous, device)
                new_ids = acquire(previous_model, train, labeled, needed, method, args,
                                  eval_tf, device, seed, i).tolist()
                del previous_model
                if str(device).startswith('cuda'): torch.cuda.empty_cache()
            atomic_json(new_ids, query_file)
        if len(new_ids) != needed or len(set(new_ids)) != len(new_ids):
            raise AssertionError('Wrong number of acquired samples or duplicated acquisition')
        if set(new_ids).intersection(labeled):
            raise AssertionError('Already-labeled sample acquired again')
        if any(j < 0 or j >= len(train) for j in new_ids):
            raise AssertionError('Out-of-range acquired sample')
        updated = labeled + new_ids
        print(f'\n[{method} seed={seed}] R{i:02d}, {pct}% labels, '
              f'count={len(updated)} (+{needed}), pool={len(train)-len(updated)}', flush=True)
        round_seed = int(seed) + 300003  # exactly same random init across rounds and methods
        seed_everything(round_seed, args.deterministic)
        model = make_model(args.backend, pretrained=args.pretrained and not (round_dir/'last.pt').exists()).to(device)
        # If checkpoint exists from interrupted training, model loads it in train_round.
        tic = time.perf_counter()
        train_ds = LabeledDataset(train, updated, train_tf)
        val_ds = LabeledDataset(val, np.arange(len(val)), eval_tf)
        train_info = train_round(model, train_ds, val_ds, args, round_dir, device, round_seed)
        val_info = evaluate_and_save(model, val, eval_tf, args, device, round_dir, 'val', seed)
        elapsed = time.perf_counter() - tic
        row = {'method': method, 'seed': seed, 'round': i, 'label_pct': pct,
               'labeled_count': len(updated), 'acquired': len(new_ids),
               'training_seconds': round(elapsed, 2),
               'best_epoch': train_info['best_epoch'],
               **{f'val_{k}': v for k, v in val_info.items()}}
        if i == len(schedule)-1:
            test_info = evaluate_and_save(model, test, eval_tf, args, device, round_dir, 'test', seed)
            row.update({f'test_{k}': v for k, v in test_info.items()})
        # Store revealed query labels only AFTER acquisition (never available to query methods).
        queried = [{'index': k, 'path': train.paths[k], 'revealed_label': int(train.labels[k])}
                   for k in new_ids]
        csv_write(round_dir/'queried_samples.csv', queried)
        history.append(row)
        labeled = updated
        atomic_json({'config': settings, 'method': method, 'seed': seed,
                     'initial_indices': initial, 'labeled_indices': labeled,
                     'completed_round': i, 'history': history}, state_file)
        csv_write(output_dir/'history.csv', history)
        if args.keep_checkpoints == 'latest' and i > 0:
            (output_dir/f'round_{i-1:02d}'/'best.pt').unlink(missing_ok=True)
        print(f"  val top1={val_info['top1']:.2f} top2={val_info['top2']:.2f} "
              f"macro_f1={val_info['macro_f1']:.2f} MAPE={val_info['clean_mape_soft']:.3f}%", flush=True)
        del model
        if str(device).startswith('cuda'): torch.cuda.empty_cache()
    if completed >= limit-1:
        print(f'[{method} seed={seed}] Already completed through requested limit.')


def run_full(seed, train, val, test, args, device, train_tf, eval_tf, output_root):
    full_dir = output_root/'full_supervised'/f'seed_{seed}'
    full_dir.mkdir(parents=True, exist_ok=True)
    state_file = full_dir/'result.json'
    if state_file.exists():
        existing = json.loads(state_file.read_text())
        if existing['config'] != config_key(args, args.data_root):
            raise RuntimeError('Full supervised config changed; use another output path')
        print(f'[full_supervised seed={seed}] Completed, skip')
        return
    print(f'\n[full_supervised seed={seed}] 100%, n={len(train)}', flush=True)
    round_seed = int(seed)+300003
    seed_everything(round_seed, args.deterministic)
    model = make_model(args.backend, pretrained=args.pretrained and not (full_dir/'last.pt').exists()).to(device)
    t = time.perf_counter()
    info = train_round(model,
                       LabeledDataset(train, np.arange(len(train)), train_tf),
                       LabeledDataset(val, np.arange(len(val)), eval_tf),
                       args, full_dir, device, round_seed)
    val_info = evaluate_and_save(model, val, eval_tf, args, device, full_dir, 'val', seed)
    test_info = evaluate_and_save(model, test, eval_tf, args, device, full_dir, 'test', seed)
    row = {'method': 'full_supervised', 'seed': seed, 'round': 0,
           'label_pct': 100, 'labeled_count': len(train), 'acquired': len(train),
           'training_seconds': round(time.perf_counter()-t,2),
           'best_epoch': info['best_epoch'],
           **{f'val_{k}': v for k,v in val_info.items()},
           **{f'test_{k}': v for k,v in test_info.items()}}
    atomic_json({'config': config_key(args, args.data_root), 'metrics': row}, state_file)
    csv_write(full_dir/'history.csv', [row])
    del model
    if str(device).startswith('cuda'): torch.cuda.empty_cache()


def main():
    args = args_parser()
    if args.device == 'auto': args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    data_root = Path(args.data_root).expanduser().resolve()
    args.data_root = str(data_root)
    print('Data root:', data_root, 'Device:', device, 'Backend:', args.backend)
    print('No BYOL. ImageNet supervised pretrained:', args.pretrained)
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.pool_batch_size < 1:
        raise ValueError('Epochs, patience and batch sizes must be positive')
    if args.single_pct is not None:
        if args.continue_from_pct is not None and not (1 <= args.continue_from_pct < args.single_pct):
            raise ValueError('--continue-from-pct must be smaller than --single-pct')
        if not (1 <= args.single_pct <= 100):
            raise ValueError('--single-pct must be between 1 and 100')
        if not (1 <= args.single_initial_pct <= args.single_pct):
            raise ValueError('--single-initial-pct must be between 1 and --single-pct')
        if args.max_rounds is not None:
            raise ValueError('--max-rounds is only for sequential AL mode, not --single-pct')
    train = read_manifest(args.data_root, 'train')
    val = read_manifest(args.data_root, 'val')
    test = read_manifest(args.data_root, 'test')
    verify_manifests(train, val, test, check_files=not args.skip_file_check and not args.dry_run)
    if args.single_pct is not None:
        count = round(len(train) * args.single_pct / 100)
        initial_count = round(len(train) * args.single_initial_pct / 100)
        if not (1 <= initial_count <= count):
            raise ValueError('Rounded initial label count must be >=1 and <= target count')
        print(f'Train={len(train)}, val={len(val)}, test={len(test)}')
        print(f'SINGLE MODE: {args.single_pct}% = {count} labeled images, '
              f'{args.single_initial_pct}% initial = {initial_count}; '
              f'methods={args.methods}; seeds={args.seeds}')
        if args.continue_from_pct is not None:
            print(f'PROGRESSIVE MODE: target {args.single_pct}%, '
                  f'continue from completed {args.continue_from_pct}%.')
        elif any(m != 'random' for m in args.methods) and count > initial_count:
            print('Note: uncertainty/hybrid methods train one shared '
                  f'{args.single_initial_pct}% warmup model per seed before the target model.')
        if args.dry_run:
            print('DRY RUN: no training started. (Image existence check skipped.)')
            return
        if args.skip_file_check:
            print('WARNING: Image file checks skipped; loader may fail if paths are missing.')
        train_tf, eval_tf = make_transforms(args.image_size)
        output_root = Path(args.output).expanduser().resolve()/f'single_{args.single_pct}pct'
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_json({'mode':'single', 'data_root': args.data_root,
                     'split_sizes': {'train':len(train),'val':len(val),'test':len(test)},
                     'settings': single_settings(args), 'methods':args.methods,
                     'seeds': args.seeds, 'label_pct': args.single_pct,
                     'labeled_count': count, 'initial_count': initial_count},
                    output_root/'experiment.json')
        for seed in args.seeds:
            if args.include_full:
                run_full(seed, train, val, test, args, device, train_tf, eval_tf, output_root)
            for method in args.methods:
                run_single(method, seed, train, val, test, args, device, train_tf,
                           eval_tf, output_root)
        print('\nFinished single-budget experiment. Run: python src/plot_al.py --input', output_root)
        return
    schedule = budgets(len(train), args.start_pct, args.max_pct, args.step_pct)
    print(f'Train={len(train)}, val={len(val)}, test={len(test)}, rounds={len(schedule)}')
    print('Budget:', ' '.join(f'R{i}:{pct}%={count}' for i,(pct,count) in enumerate(schedule)))
    if args.dry_run:
        print('DRY RUN: no training started. (Image existence check skipped.)')
        return
    if args.skip_file_check:
        print('WARNING: Image file checks skipped; loader may fail if paths are missing.')
    train_tf, eval_tf = make_transforms(args.image_size)
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json({'data_root': args.data_root, 'split_sizes': {'train': len(train),
                 'val':len(val),'test':len(test)}, 'settings': config_key(args,args.data_root),
                 'methods': args.methods, 'seeds': args.seeds,
                 'schedule': [{'round': i, 'label_pct': pct, 'n': n}
                              for i,(pct,n) in enumerate(schedule)]}, output_root/'experiment.json')
    for seed in args.seeds:
        if args.include_full:
            run_full(seed, train, val, test, args, device, train_tf, eval_tf, output_root)
        for method in args.methods:
            run_al(method, seed, train, val, test, args, device, train_tf, eval_tf, schedule, output_root)
    print('\nFinished requested experiments. Run: python src/plot_al.py --input', output_root)


if __name__ == '__main__':
    main()