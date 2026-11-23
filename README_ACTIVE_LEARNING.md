# GrainSet-Rice — ResNet50 Active Learning (2%–50%)

**Ready-to-run training source code** added to the uploaded GrainSet repository.
No BYOL, pseudo-labeling, or self-supervised pretraining.

## Implemented protocol

| Item | Setting |
|---|---|
| Classification | GrainSet-Rice, 8 classes, ORIGINAL `src/models/model.py` (timm ResNet50) |
| Train/val/test | 24,767 / 3,095 / 3,100 **images** (original split files) |
| Initial labels | 2% = 495 images, identical across methods within a seed |
| Acquisition | Additional 2% per round, cumulative budget `2, 4, ..., 50%` |
| Rounds | R0–R24, 25 evaluation points, 24 acquisitions |
| Final labeled subset | 50% = 12,384 images |
| Strategies | `random`, `least_confidence`, `entropy`, `margin`, `hybrid` |
| Hybrid | Top-3×batch uncertainty filtering, 64D random projection + cosine farthest-first diversity |
| Main backbone | **Existing ResNet50**, initialized with *supervised ImageNet* weights if available |
| Training | RAdam, cosine scheduler, LR 0.0002, batch 32, max 50 epochs, patience 10 |
| Model selection | Highest **validation macro-F1**, no test-set optimization |
| Repetitions | seeds 42, 123, 509 |
| Reference | ResNet50, 100% train labels (`--include-full`) |
| Validation per round | Top-1, Top-2, macro-F1, balanced accuracy, recall IM, synthetic-bag composition MAPE/MAE |
| Test | Evaluated at **50% label budget** and for 100% supervised reference only |
| Restart | Resume last incomplete epoch and/or next incomplete AL round |

## 1. Install

Requires Python >=3.10, CUDA-compatible PyTorch, and image files from **GrainSet-Rice**.
The uploaded `grainset.zip` contains the **code and datalist text files only**, not images.

```bash
# Example after extracting this ZIP
cd grainset_active_learning

# Install CUDA-enabled PyTorch per https://pytorch.org/get-started/locally/
# Example: only if compatible with the installed NVIDIA driver and CUDA runtime
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements-al.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

For main experiments the `timm` package is **required**; the original repository's `src/models/model.py` imports it. The alternative `--backend torchvision` is intended for debugging, and is **not identical** to the main original-model experiment.

## 2. Image directory expected

Set `--data-root` to the actual extracted Rice image folder. The default command assumes `/datasets/rice`.

```text
/datasets/rice/
    train/
        0_NOR/Grainset_rice_*.png
        1_.../
        ...
    test/
        ...

project-root/
    datasets/datalist/rice_train.txt
    datasets/datalist/rice_val.txt
    datasets/datalist/rice_test.txt
```

GrainSet's **validation manifest refers to images beneath `train/`**, not necessarily to a separate `val/` directory. Do not randomly re-split the data for the primary experiment.

The program checks that all listed image files exist, that there are no identical paths between splits, and that labels are in `[0,7]` before training.

## 3. Check budget without training

```bash
python src/train_al.py --data-root /datasets/rice --dry-run
```

Expected: R0 at 2% (495), R24 at 50% (12,384).

## 4. Short experiment first

This command runs only R0–R2 **but preserves the 2–50% schedule** for resuming with the same output later:

```bash
python src/train_al.py \
  --data-root /datasets/rice \
  --output runs/active_learning \
  --methods random entropy hybrid \
  --seeds 42 \
  --max-rounds 3 \
  --backend timm --pretrained
```

The script only executes the first 3 rounds, 2%, 4%, 6%. To continue, run again **without** `--max-rounds 3` and with the exact same training hyperparameters. For all 5 acquisition methods and 3 seeds, use the next command.

## 5. COMPLETE 2%–50% experiment

```bash
DATA_ROOT=/datasets/rice bash scripts/run_al.sh
```

Equivalent explicit command:

```bash
python src/train_al.py \
  --data-root /datasets/rice \
  --output runs/active_learning \
  --methods random least_confidence entropy margin hybrid \
  --seeds 42 123 509 \
  --start-pct 2 --step-pct 2 --max-pct 50 \
  --batch-size 32 --pool-batch-size 128 --workers 4 \
  --epochs 50 --patience 10 --lr 2e-4 \
  --backend timm --pretrained --include-full

python src/plot_al.py --input runs/active_learning
```

This is **375 Active Learning fits + 3 full-supervised fits** and can be very compute-intensive. `--max-rounds 3` and one seed are recommended to verify the local environment before the full run.

To fit on a smaller GPU, lower `--batch-size` and/or `--pool-batch-size` (changing parameters for a previously started run is deliberately rejected: use a new output directory).

`--no-pretrained` turns off ImageNet weights (model trained from random initialization); it does **not** enable BYOL. Pretrained ImageNet weights may need to be downloaded on first launch unless already cached.

## 6. Strategies

- **Random**: query uniform random unseen images.
- **Least Confidence**: query smallest max softmax probability.
- **Entropy**: query highest Shannon entropy.
- **Margin**: query smallest gap between the two largest probabilities.
- **Hybrid**: first rank all unlabeled samples by entropy; from top `3 × acquisition_size`, choose diverse samples using ResNet50 penultimate-layer embeddings with greedy farthest-first selection.

Only image tensors and model predictions enter acquisition. True training-pool labels are revealed **after** image indices have been selected. All five methods use identical R0 samples (same seed).

## 7. MAPE — very important interpretation

The classifier predicts **individual rice-grain category**, not a continuous purity score. As such, it is **not meaningful** to directly calculate MAPE on Top-1 labels.

To evaluate composition, the code synthesizes *bags* from `val` (each round) or `test` (final 50% / 100% baseline). Each bag samples held-out grain images without replacement *within a bag*. Separate bags may reuse the same images and are therefore not fully independent.

Default: 100/200/500 grains × 1%/2%/5%/10% impurity × 30 repeated bags per combination. The target is **count-based proportion of non-impurity grains**, not mass-based rice quality or varietal purity:

`clean% = 100 × (1 − (# grains from class 7 IM / # objects))`.

Two estimation methods:

1. **Hard**: count class-7 predictions (Top-1).
2. **Soft**: average predicted class-7 probabilities.

Output:
- `val_clean_mape_soft`, `val_clean_mape_hard`: MAPE of *clean percentage*, percentage units.
- `val_impurity_mae_pp_soft`, `val_impurity_mae_pp_hard`: absolute error in **percentage points** of impurity proportion.
- `val_normal_mae_pp_soft`, `val_normal_mae_pp_hard`: absolute error in normal proportion.
- `val_always_clean_mape`: trivial model always predicting 100% clean. Compare against it to avoid misleading low MAPE when bags are mostly clean.

For real commercial purity claims you need physically measured mixed-sample annotations, preferably by mass if quality standards require mass fractions. Synthetic bags only support **simulation**. Composition predictions may be biased without probability calibration; calibration is an optional follow-on ablation not applied here.

The test split is kept out of training/model selection and used only at the 50% endpoint and 100% reference. Report validation curves at 2–50% and test results only for held-out endpoint(s).

## 8. Result structure

```text
runs/active_learning/
    experiment.json
    random/
        seed_42/
            state.json                  # saved labeled pool and completed round
            history.csv                 # validation metric per round, final test at 50%
            round_00/
                queried_indices.json   # only added indices (0 for R0)
                best.pt                 # most recent checkpoint, default pruning policy
                val_predictions.npz    # paths, y_true, probabilities, y_pred
                val_confusion.csv
                val_metrics.json
                queried_samples.csv
            round_01/... 
            ...
            round_24/
                best.pt
                test_predictions.npz
                test_confusion.csv
                test_metrics.json
    entropy/...
    hybrid/...
    full_supervised/seed_42/...
    all_runs.csv
    summary_mean_std.csv
    aulc_by_seed.csv
    figures/
        val_top1.png / .pdf
        val_top2.png / .pdf
        val_macro_f1.png / .pdf
        val_clean_mape_soft.png / .pdf
        val_impurity_mae_pp_soft.png / .pdf
```

The default `--keep-checkpoints latest` deletes OLD `best.pt` files once a newer round has been successfully persisted. This saves storage. `--keep-checkpoints all` retains every round's best weights (many GB); note that **the default cannot restore old intermediate weights**, but all metrics, predictions, and query lists remain saved.

## 9. Resume after interruption

Simply rerun the same command with identical model/training settings. The code validates per-run configuration and restores the last unfinished training epoch when a `last.pt` exists, or resumes at the next not-yet-completed AL round.

Do not change `--epochs`, `--lr`, `--image-size`, `--pretrained`, `--start-pct`, `--max-pct`, etc. inside an existing output directory. Instead change `--output`. `--max-rounds` may be changed to extend a short run because it does not change the long-term experimental budget.

## 10. Tests and verification

```bash
python -m unittest discover -s tests -v
python src/train_al.py --data-root /datasets/rice --dry-run
```

Validation performed when creating this release:
- 6 unit tests passed.
- Datalist parsing and full 2–50% budget schedule validated against uploaded rice manifests.
- End-to-end **synthetic image** smoke test passed with Torchvision ResNet50 (`--backend torchvision --no-pretrained`) over Random and Hybrid acquisition, including resume-capable checkpoints, per-round metrics, final-test outputs, and visualization.

**NOT trained on real GrainSet rice photos here.** Real-image training must be run in your GPU environment. The timm backend could not be run in the creation environment because timm was not installed and downloading packages was unavailable; its factory directly calls the repository's original model function.

## 11. Methodological notes

1. **Legacy augmentation is replaced deliberately** with RGB, ImageNet normalization, RandomResizedCrop, flips and mild brightness/contrast adjustments. The uploaded repository reads BGR + dataset-specific BGR normalization. ImageNet normalization is appropriate when using ImageNet-pretrained weights. All five AL methods and the full-supervised reference share exactly the same preprocessing; do not directly compare scores with earlier runs that used different preprocessing.
2. Initial labeled samples are **random, not stratified by hidden labels**. Some minorities may be poorly represented at R0; this is part of the true low-label challenge. The same R0 is reused across methods for every seed.
3. Existing GrainSet image filenames sometimes share acquisition timestamps across official splits. An independent group-aware split would be a useful additional evaluation if image provenance can be established. This code uses the official split and detects exact-file overlap, **not** physical-specimen leakage.
4. The target Top-1 ≥95%, Top-2 ≥98%, clean MAPE <2% are research goals, **not guaranteed outcomes**.
5. Model/strategy selection uses validation macro-F1. The 50% test result is held out and should not be used to tune the acquisition strategy.
