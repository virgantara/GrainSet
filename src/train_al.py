#!/usr/bin/env python3
"""GrainSet Rice / ResNet50 / pool-based active learning, budgets 2..50%.

Main experiments: Random, LC, Entropy, Margin, Entropy->Diversity.
Uses original src/models/model.py (timm ResNet50), no BYOL.
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
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 509])
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
    p.add_argument('--workers', type=int, default=4)
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
    return p.parse_args()


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
    train = read_manifest(args.data_root, 'train')
    val = read_manifest(args.data_root, 'val')
    test = read_manifest(args.data_root, 'test')
    verify_manifests(train, val, test, check_files=not args.skip_file_check and not args.dry_run)
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
