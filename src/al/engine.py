"""ResNet50 factory, training, inference, metrics and atomic checkpoints."""
from __future__ import annotations

from pathlib import Path
import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score


def seed_everything(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_model(backend='timm', pretrained=True, num_classes=8):
    if backend == 'timm':
        # Unmodified ORIGINAL src/models/model.py -> timm.models.resnet.resnet50
        # Passing pretrained=True is supported through **kwargs.
        from types import SimpleNamespace
        from models.model import get_model
        cfg = SimpleNamespace(MODEL=SimpleNamespace(NAME='resnet50', RESUME_PATH=''))
        return get_model(cfg, num_classes=num_classes, pretrained=pretrained)
    if backend == 'torchvision':
        # Debug fallback: same ResNet50 family, NOT the repo's timm backend.
        from torchvision.models import resnet50, ResNet50_Weights
        return _tv_pretrained(num_classes) if pretrained else resnet50(weights=None, num_classes=num_classes)
    raise ValueError(f"Unknown backend: {backend}")


def _tv_pretrained(num_classes):
    from torchvision.models import resnet50, ResNet50_Weights
    m = resnet50(weights=ResNet50_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def forward_model(model, images, want_features=False):
    if hasattr(model, 'forward_features') and hasattr(model, 'forward_head'):
        z = model.forward_features(images)
        logits = model.forward_head(z)
        if not want_features:
            return logits, None
        features = model.forward_head(z, pre_logits=True)
        return logits, features.flatten(1)
    # Torchvision debug backend uses hook on avgpool to extract embeddings.
    if not want_features:
        return model(images), None
    feats = []
    hook = model.avgpool.register_forward_hook(lambda _mod, _in, out: feats.append(out))
    try:
        logits = model(images)
    finally:
        hook.remove()
    return logits, feats[0].flatten(1)


def make_loader(dataset, batch_size, workers, shuffle=False, seed=0, device='cpu'):
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=str(device).startswith('cuda'), drop_last=False,
                      generator=gen, persistent_workers=workers > 0)


def classify(y, prob):
    y = np.asarray(y, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float32)
    pred = prob.argmax(1)
    top2 = np.argsort(prob, axis=1)[:, -2:]
    metric = {
        'top1': float(100 * accuracy_score(y, pred)),
        'top2': float(100 * np.mean((top2 == y[:,None]).any(axis=1))),
        'macro_f1': float(100 * f1_score(y, pred, labels=list(range(8)), average='macro', zero_division=0)),
        'balanced_accuracy': float(100 * balanced_accuracy_score(y, pred)),
        'recall_im': float(100 * recall_score(y == 7, pred == 7, zero_division=0)),
    }
    return metric, confusion_matrix(y, pred, labels=list(range(8)))


@torch.no_grad()
def inference(model, loader, device, want_features=False):
    model.eval()
    all_probs, all_features, all_targets, all_indices = [], [], [], []
    for batch in loader:
        if len(batch) == 3:
            images, labels, indices = batch
            all_targets.extend(labels.numpy().tolist())
        else:
            images, indices = batch
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', enabled=str(device).startswith('cuda')):
            logits, features = forward_model(model, images, want_features)
        all_probs.append(logits.float().softmax(1).cpu().numpy())
        if want_features:
            all_features.append(features.float().cpu().numpy())
        all_indices.extend(indices.numpy().tolist())
    p = np.concatenate(all_probs) if all_probs else np.zeros((0,8), np.float32)
    f = np.concatenate(all_features) if want_features else None
    return np.asarray(all_indices, np.int64), np.asarray(all_targets, np.int64), p, f


def atomic_torch_save(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temp)
    os.replace(temp, path)


def load_weights(model, path, device):
    # Full run-produced state dict; not arbitrary pickle model.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    return checkpoint


def train_round(model, train_ds, val_ds, cfg, output_dir, device, round_seed):
    """Train from same initial weights every AL round; resume interrupted epochs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_path, last_path = out/'best.pt', out/'last.pt'
    optimizer = torch.optim.RAdam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=str(device).startswith('cuda'))
    criterion = nn.CrossEntropyLoss()
    start_epoch, best_f1, best_epoch, stale = 0, -1., -1, 0
    if last_path.is_file():
        ckpt = load_weights(model, last_path, device)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = int(ckpt['epoch']) + 1
        best_f1 = float(ckpt['best_f1'])
        best_epoch = int(ckpt['best_epoch'])
        stale = int(ckpt['stale'])
        print(f'  resume epoch {start_epoch+1} / {cfg.epochs}')
    best_stats = None
    for epoch in range(start_epoch, cfg.epochs):
        seed_everything(round_seed + epoch, cfg.deterministic)
        train_loader = make_loader(train_ds, cfg.batch_size, cfg.workers, shuffle=True,
                                   seed=round_seed + epoch, device=device)
        model.train(); loss_sum, n = 0., 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=str(device).startswith('cuda')):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * len(labels)
            n += len(labels)
        eval_loader = make_loader(val_ds, cfg.batch_size, cfg.workers, device=device)
        _, val_y, val_p, _ = inference(model, eval_loader, device)
        val_stats, _ = classify(val_y, val_p)
        scheduler.step()
        improved = val_stats['macro_f1'] > best_f1 + 1e-6
        if improved:
            best_f1 = val_stats['macro_f1']
            best_epoch = epoch
            stale = 0
            best_stats = val_stats
            atomic_torch_save({'model': model.state_dict(), 'epoch': epoch, 'val': val_stats}, best_path)
        else:
            stale += 1
        atomic_torch_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                           'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                           'epoch': epoch, 'best_f1': best_f1, 'best_epoch': best_epoch,
                           'stale': stale}, last_path)
        print(f"    epoch {epoch+1:02d}/{cfg.epochs} train_loss={loss_sum/max(1,n):.4f} "
              f"val_top1={val_stats['top1']:.2f} val_top2={val_stats['top2']:.2f} "
              f"val_macro_f1={val_stats['macro_f1']:.2f} best_epoch={best_epoch+1}", flush=True)
        if stale >= cfg.patience:
            print(f'    early stop at epoch {epoch+1}')
            break
    if not best_path.is_file():
        raise RuntimeError(f'Missing best checkpoint: {best_path}')
    load_weights(model, best_path, device)
    # Best run is preserved. Last is not needed once round is completed.
    last_path.unlink(missing_ok=True)
    return {'best_epoch': best_epoch + 1, 'best_val_macro_f1': best_f1, **(best_stats or {})}
