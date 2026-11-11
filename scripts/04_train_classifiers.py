#!/usr/bin/env python3
"""
Phase 3: Classifier Training on Pass W Whole-Clip EmbeddingGemma 2 Embeddings.

Plan decisions enforced:
  D3: SVM, MLP, XGBoost
  D6: GroupKFold(n_splits=6) by actor — speaker-independent evaluation
  D5: Primary = UAR (macro-recall), Secondary = macro-F1
  D9: 3 seeds per config; report mean ± std

Output: results/classifier_results.csv + results/classifier_results.txt (human-readable report)
"""

import os
import csv
import json
import warnings
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import recall_score, f1_score, confusion_matrix, classification_report

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARNING] xgboost not installed — skipping XGBoost. Install with: pip install xgboost")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_PATH = os.path.join(WORKSPACE_DIR, "embeddings", "pass_w.npz")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "classifier_results.csv")
RESULTS_TXT = os.path.join(RESULTS_DIR, "classifier_results.txt")

SEEDS = [42, 123, 7]
N_FOLDS = 6
CLASSES = ["angry", "happy", "neutral", "sad"]


def make_configs():
    """
    Returns list of (name, build_fn) tuples.
    build_fn(seed) -> sklearn Pipeline.
    """
    configs = []

    # ── SVM ──────────────────────────────────────────────────────────
    for C in [0.01, 0.1, 1.0, 10.0]:
        for kernel in ["linear", "rbf"]:
            name = f"SVM_{kernel}_C{C}"
            C_, kernel_ = C, kernel  # capture
            configs.append((
                name,
                lambda seed, C=C_, k=kernel_: Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel=k, C=C, class_weight="balanced",
                                random_state=seed, probability=False)),
                ])
            ))

    # ── MLP ──────────────────────────────────────────────────────────
    for hidden in [(256,), (512,), (256, 128), (512, 256)]:
        for lr in [1e-3, 3e-4]:
            name = f"MLP_{'x'.join(str(h) for h in hidden)}_lr{lr}"
            h_, lr_ = hidden, lr
            configs.append((
                name,
                lambda seed, h=h_, lr=lr_: Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", MLPClassifier(hidden_layer_sizes=h, learning_rate_init=lr,
                                         max_iter=500, early_stopping=True,
                                         validation_fraction=0.1, n_iter_no_change=15,
                                         random_state=seed)),
                ])
            ))

    # ── XGBoost ──────────────────────────────────────────────────────
    if HAS_XGB:
        for lr in [0.05, 0.1]:
            for n_est in [200, 400]:
                for max_depth in [4, 6]:
                    name = f"XGB_lr{lr}_n{n_est}_d{max_depth}"
                    lr_, n_, d_ = lr, n_est, max_depth
                    configs.append((
                        name,
                        lambda seed, lr=lr_, n=n_, d=d_: Pipeline([
                            ("scaler", StandardScaler()),
                            ("clf", XGBClassifier(
                                n_estimators=n, max_depth=d, learning_rate=lr,
                                eval_metric="mlogloss",
                                random_state=seed, verbosity=0,
                                n_jobs=2,
                            )),
                        ])
                    ))

    return configs


def evaluate_config(name, build_fn, X, y, actors, seeds, n_folds):
    """
    Run n_folds GroupKFold × len(seeds) independent trials.
    Returns dict of aggregated metrics.
    """
    all_uars = []
    all_f1s = []
    seed_uars = {s: [] for s in seeds}
    seed_f1s = {s: [] for s in seeds}

    gkf = GroupKFold(n_splits=n_folds)

    for seed in seeds:
        fold_uars = []
        fold_f1s = []
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=actors)):
            model = build_fn(seed)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            uar = recall_score(y[val_idx], preds, average="macro", zero_division=0)
            mf1 = f1_score(y[val_idx], preds, average="macro", zero_division=0)
            fold_uars.append(uar)
            fold_f1s.append(mf1)

        seed_uars[seed] = fold_uars
        seed_f1s[seed] = fold_f1s
        all_uars.extend(fold_uars)
        all_f1s.extend(fold_f1s)

    # Per-seed mean, then mean-of-means and std-of-means (D9)
    seed_mean_uars = [np.mean(seed_uars[s]) for s in seeds]
    seed_mean_f1s = [np.mean(seed_f1s[s]) for s in seeds]

    return {
        "config": name,
        "uar_mean": np.mean(seed_mean_uars),
        "uar_std": np.std(seed_mean_uars),
        "f1_mean": np.mean(seed_mean_f1s),
        "f1_std": np.std(seed_mean_f1s),
        "seed_uar_means": seed_mean_uars,
        "seed_f1_means": seed_mean_f1s,
    }


def main():
    print("=" * 70)
    print("PHASE 3: CLASSIFIER TRAINING ON PASS W EMBEDDINGS")
    print(f"Seeds: {SEEDS} | Folds: {N_FOLDS} | Classes: {CLASSES}")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load embeddings
    print(f"\nLoading embeddings from {NPZ_PATH}...")
    data = np.load(NPZ_PATH)
    X = data["embeddings"].astype(np.float32)
    raw_labels = data["labels"]
    actors = data["actors"]
    print(f"X: {X.shape}, labels: {raw_labels.shape}, unique actors: {len(np.unique(actors))}")
    print(f"Class distribution: { {c: int((raw_labels == c).sum()) for c in CLASSES} }")

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y = np.array([class_to_idx[c] for c in raw_labels], dtype=np.int32)

    configs = make_configs()
    total = len(configs)
    print(f"\nTotal configurations to evaluate: {total}")
    print(f"Each: {N_FOLDS} folds × {len(SEEDS)} seeds = {N_FOLDS * len(SEEDS)} runs per config\n")

    all_results = []
    best_uar = 0.0
    best_config = None

    # Open CSV for streaming writes
    csv_fields = ["config", "uar_mean", "uar_std", "f1_mean", "f1_std",
                  "seed42_uar", "seed123_uar", "seed7_uar",
                  "seed42_f1", "seed123_f1", "seed7_f1"]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
        writer.writeheader()

        for idx, (name, build_fn) in enumerate(configs, 1):
            print(f"[{idx:02d}/{total}] {name} ...", end=" ", flush=True)
            try:
                res = evaluate_config(name, build_fn, X, y, actors, SEEDS, N_FOLDS)
                uar_m, uar_s = res["uar_mean"] * 100, res["uar_std"] * 100
                f1_m, f1_s = res["f1_mean"] * 100, res["f1_std"] * 100
                seed_uars = [u * 100 for u in res["seed_uar_means"]]
                seed_f1s = [f * 100 for f in res["seed_f1_means"]]
                print(f"UAR={uar_m:.2f}%±{uar_s:.2f}  F1={f1_m:.2f}%±{f1_s:.2f}")

                row = {
                    "config": name,
                    "uar_mean": f"{uar_m:.4f}",
                    "uar_std": f"{uar_s:.4f}",
                    "f1_mean": f"{f1_m:.4f}",
                    "f1_std": f"{f1_s:.4f}",
                    "seed42_uar": f"{seed_uars[0]:.4f}",
                    "seed123_uar": f"{seed_uars[1]:.4f}",
                    "seed7_uar": f"{seed_uars[2]:.4f}",
                    "seed42_f1": f"{seed_f1s[0]:.4f}",
                    "seed123_f1": f"{seed_f1s[1]:.4f}",
                    "seed7_f1": f"{seed_f1s[2]:.4f}",
                }
                writer.writerow(row)
                csv_f.flush()
                all_results.append(res)

                if uar_m > best_uar:
                    best_uar = uar_m
                    best_config = name

            except Exception as e:
                print(f"ERROR: {e}")

    # Sort by UAR descending
    all_results.sort(key=lambda r: r["uar_mean"], reverse=True)

    # Write human-readable report
    print("\n" + "=" * 70)
    print("FINAL RESULTS (sorted by UAR, mean over 3 seeds × 6 folds)")
    print(f"{'Rank':<5} {'Config':<40} {'UAR (%)':>10} {'±':>5} {'F1 (%)':>10} {'±':>5}")
    print("-" * 70)
    lines = []
    lines.append(f"{'Rank':<5} {'Config':<40} {'UAR (%)':>10} {'±':>5} {'F1 (%)':>10} {'±':>5}")
    lines.append("-" * 70)

    for rank, res in enumerate(all_results, 1):
        uar_m = res["uar_mean"] * 100
        uar_s = res["uar_std"] * 100
        f1_m = res["f1_mean"] * 100
        f1_s = res["f1_std"] * 100
        line = f"{rank:<5} {res['config']:<40} {uar_m:>10.2f} {uar_s:>5.2f} {f1_m:>10.2f} {f1_s:>5.2f}"
        print(line)
        lines.append(line)

    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write("PHASE 3 CLASSIFIER RESULTS — EmbeddingGemma 2 Pass W (Whole-Clip)\n")
        f.write(f"Seeds: {SEEDS} | Folds: {N_FOLDS} | GroupKFold by actor\n")
        f.write(f"Baselines: Random=25.0%  Majority=33.3%\n\n")
        f.write("\n".join(lines))
        f.write(f"\n\nBest overall config: {best_config} with UAR={best_uar:.2f}%\n\n")

        # Family winners and detailed evaluation
        f.write("=" * 70 + "\n")
        f.write("BEST MODEL PER FAMILY & DETAILED EVALUATION\n")
        f.write("=" * 70 + "\n\n")

        config_dict = {c[0]: c[1] for c in configs}
        for family in ["SVM", "MLP", "XGB"]:
            fam_results = [r for r in all_results if r["config"].startswith(family)]
            if fam_results:
                best_fam = fam_results[0]
                b_name = best_fam["config"]
                b_fn = config_dict[b_name]
                f.write(f"--- Best {family}: {b_name} ---\n")
                f.write(f"UAR: {best_fam['uar_mean']*100:.2f}% ± {best_fam['uar_std']*100:.2f}%\n")
                f.write(f"Macro-F1: {best_fam['f1_mean']*100:.2f}% ± {best_fam['f1_std']*100:.2f}%\n")
                f.write(f"Per-seed UARs: {[round(u*100, 2) for u in best_fam['seed_uar_means']]}\n")

                # Compute out-of-fold metrics on best seed
                gkf = GroupKFold(n_splits=N_FOLDS)
                best_s = None
                best_s_uar = -1.0
                best_s_preds = np.zeros(len(y), dtype=np.int32)
                for s_idx, s in enumerate(SEEDS):
                    s_preds = np.zeros(len(y), dtype=np.int32)
                    for tr_idx, val_idx in gkf.split(X, y, groups=actors):
                        m = b_fn(s)
                        m.fit(X[tr_idx], y[tr_idx])
                        s_preds[val_idx] = m.predict(X[val_idx])
                    s_uar = recall_score(y, s_preds, average="macro", zero_division=0)
                    if s_uar > best_s_uar:
                        best_s_uar = s_uar
                        best_s = s
                        best_s_preds = s_preds

                cm = confusion_matrix(y, best_s_preds)
                cr = classification_report(y, best_s_preds, target_names=CLASSES, digits=4)
                f.write(f"Best Seed: {best_s} (Pooled OOF UAR: {best_s_uar*100:.2f}%)\n")
                f.write("Classification Report:\n" + cr + "\n")
                f.write("Confusion Matrix (rows=true, cols=pred):\n")
                f.write(f"{'':>10}" + "".join(f"{c:>10}" for c in CLASSES) + "\n")
                for r_idx, r_name in enumerate(CLASSES):
                    f.write(f"{r_name:>10}" + "".join(f"{cm[r_idx, c_idx]:>10}" for c_idx in range(len(CLASSES))) + "\n")
                f.write("\n" + "-" * 50 + "\n\n")

    print("=" * 70)
    print(f"\nBest config: {best_config} → UAR = {best_uar:.2f}%")
    print(f"\nResults saved to:\n  {RESULTS_CSV}\n  {RESULTS_TXT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
