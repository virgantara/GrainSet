"""Simulated bag-level composition. Counts of objects, NOT mass/varietal purity."""
from __future__ import annotations
import numpy as np


def bag_metrics(y_true, prob, *, bag_sizes=(100, 200, 500), rates=(0.01, 0.02, 0.05, 0.10),
                repetitions=30, seed=2026, impurity_class=7, normal_class=0) -> dict:
    """Purity=100*(1 - N_impurity/N); bags sampled from held-out split only.

    Uses fixed seeded sampling for paired comparisons across AL methods.
    A bag has distinct images; different bags may reuse images.
    """
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(prob, dtype=np.float64)
    if len(y) != len(p) or p.ndim != 2 or len(y) == 0:
        raise ValueError("Invalid labels/probability shapes")
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    rng = np.random.default_rng(seed)
    imp = np.flatnonzero(y == impurity_class)
    non = np.flatnonzero(y != impurity_class)
    hard = p.argmax(axis=1)
    results = {name: [] for name in (
        'clean_mape_hard', 'clean_mape_soft', 'clean_mae_pp_hard', 'clean_mae_pp_soft',
        'impurity_mae_pp_hard', 'impurity_mae_pp_soft',
        'normal_mae_pp_hard', 'normal_mae_pp_soft',
        'always_clean_mape')}
    used = 0
    for size in bag_sizes:
        for rate in rates:
            n_imp = round(int(size) * float(rate))
            n_non = int(size) - n_imp
            if n_imp > len(imp) or n_non > len(non):
                raise ValueError(f"Not enough held-out samples for bag_size={size}, impurity_rate={rate}")
            for _ in range(repetitions):
                picks = np.concatenate((rng.choice(imp, n_imp, replace=False),
                                        rng.choice(non, n_non, replace=False)))
                true_clean = 100. * n_non / size
                true_normal = 100. * np.mean(y[picks] == normal_class)
                clean_hard = 100. * np.mean(hard[picks] != impurity_class)
                clean_soft = 100. * (1 - p[picks, impurity_class].mean())
                norm_hard = 100. * np.mean(hard[picks] == normal_class)
                norm_soft = 100. * p[picks, normal_class].mean()
                for k, val in [('clean_mape_hard', 100*abs(clean_hard-true_clean)/true_clean),
                               ('clean_mape_soft', 100*abs(clean_soft-true_clean)/true_clean),
                               ('clean_mae_pp_hard', abs(clean_hard-true_clean)),
                               ('clean_mae_pp_soft', abs(clean_soft-true_clean)),
                               ('impurity_mae_pp_hard', abs(clean_hard-true_clean)),
                               ('impurity_mae_pp_soft', abs(clean_soft-true_clean)),
                               ('normal_mae_pp_hard', abs(norm_hard-true_normal)),
                               ('normal_mae_pp_soft', abs(norm_soft-true_normal)),
                               ('always_clean_mape', 100*abs(100-true_clean)/true_clean)]:
                    results[k].append(val)
                used += 1
    return {'bags': used, **{k: float(np.mean(v)) for k, v in results.items()}}
