#!/usr/bin/env python3
"""Aggregate seed means/std and produce journal-ready learning curves from validation."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='runs/active_learning')
    args = ap.parse_args()
    root = Path(args.input)
    histories = sorted(root.glob('*/seed_*/history.csv'))
    if not histories:
        raise SystemExit(f'No history.csv found under {root}')
    all_data = pd.concat([pd.read_csv(p) for p in histories], ignore_index=True)
    all_data.to_csv(root/'all_runs.csv', index=False)
    numeric = [c for c in all_data.columns if c.startswith('val_') or c.startswith('test_')]
    agg = all_data.groupby(['method','label_pct'], as_index=False)[numeric].agg(['mean','std'])
    agg.columns = ['_'.join(v).strip('_') for v in agg.columns.to_flat_index()]
    agg.to_csv(root/'summary_mean_std.csv', index=False)
    plots = root/'figures'; plots.mkdir(exist_ok=True)
    for metric in ('val_top1','val_top2','val_macro_f1','val_clean_mape_soft',
                   'val_impurity_mae_pp_soft'):
        plt.figure(figsize=(8,5))
        for method, grp in all_data[all_data.method != 'full_supervised'].groupby('method'):
            stat = grp.groupby('label_pct')[metric].agg(['mean','std']).sort_index()
            plt.plot(stat.index, stat['mean'], marker='.', label=method)
            if stat['std'].notna().any():
                lo = stat['mean']-stat['std'].fillna(0)
                hi = stat['mean']+stat['std'].fillna(0)
                plt.fill_between(stat.index.to_numpy(), lo.to_numpy(), hi.to_numpy(), alpha=.12)
        if (all_data.method == 'full_supervised').any():
            baseline = all_data.loc[all_data.method=='full_supervised',metric].mean()
            plt.axhline(baseline, ls='--', color='k', label='Full supervised (100%)')
        plt.xlabel('Labeled training pool (%)'); plt.ylabel(metric)
        plt.title('GrainSet-Rice / ResNet50 / Active Learning')
        plt.grid(alpha=.2); plt.legend()
        plt.tight_layout()
        plt.savefig(plots/f'{metric}.png', dpi=200)
        plt.savefig(plots/f'{metric}.pdf')
        plt.close()
    # Normalized area under the learning curve, averaged across seeds.
    aulc_rows = []
    import numpy as np
    for (method, seed), group in all_data[all_data.method!='full_supervised'].groupby(['method','seed']):
        group = group.sort_values('label_pct')
        if len(group)<2: continue
        for metric in ('val_top1','val_top2','val_macro_f1','val_clean_mape_soft'):
            x=group.label_pct.to_numpy(dtype=float)
            y=group[metric].to_numpy(dtype=float)
            aulc_rows.append({'method': method, 'seed': seed, 'metric': metric,
                              'start_pct': x[0], 'end_pct': x[-1],
                              'aulc': float(np.trapezoid(y,x)/(x[-1]-x[0]))})
    pd.DataFrame(aulc_rows).to_csv(root/'aulc_by_seed.csv', index=False)
    print('Saved:', root/'summary_mean_std.csv', root/'aulc_by_seed.csv', plots)


if __name__=='__main__':
    main()
