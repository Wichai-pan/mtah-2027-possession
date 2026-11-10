"""Score predictions with the official metric, on one match or many.

Two aggregates over several matches:
  mean    the average of per-match scores (each match weighs the same)
  pooled  all matches concatenated (with X separators) and scored once -- closest to how a single
          full match is scored, and less noisy than the mean for spellF1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .official import possession_metrics as pm
from .schema import X

KEYS = ["spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells",
        "mean_spell_duration_s", "gt_mean_spell_duration_s", "alive_dead_f1", "team_acc_gt_alive"]


def score(gt: np.ndarray, pred: np.ndarray) -> dict:
    r = pm.evaluate(np.asarray(gt, np.int8), np.asarray(pred, np.int8))
    out = {k: r[k] for k in KEYS}
    for c in "HAD":
        out[f"f1_{c}"] = r["per_class"][c]["f1"]
    return out


def score_many(gts: list[np.ndarray], preds: list[np.ndarray], names: list[str]) -> pd.DataFrame:
    rows = [{"match": n, **score(g, p)} for g, p, n in zip(gts, preds, names)]
    df = pd.DataFrame(rows)
    mean = df.drop(columns="match").mean(numeric_only=True).to_dict()
    sep = np.full(10, X, np.int8)
    pooled_gt = np.concatenate([x for g in gts for x in (g, sep)])
    pooled_pred = np.concatenate([x for p in preds for x in (p, sep)])
    pooled = score(pooled_gt, pooled_pred)
    df = pd.concat([df, pd.DataFrame([{"match": "MEAN", **mean}, {"match": "POOLED", **pooled}])],
                   ignore_index=True)
    return df
