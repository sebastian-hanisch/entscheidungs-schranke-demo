"""Misst die Bänder, die die App als "vorgerechnet" zeigt, und schreibt `esc_bands.json` (feste Zufalls-Seeds 0 bis 29).

    python tools/measure_bands.py [--out esc_bands.json] [--seeds 30]

Inhalt je Zelle (Grad 1 / 4 / 6 x Rauschstufe 0,25 / 0,5 / 0,75; KQ-Modell, 1000 Trainings-, 1000 Kalibrier-, 5000 Testfälle, Ziel-Überdeckung 0,9):
je Seed Überdeckung und Breite der drei Schranken (Summe der Kantenschranken A, mit Bonferroni B, konform auf den Wegkosten C), der Optimismus des gewählten Weges und, als
Kontrolle, der Optimismus eines zufällig gewählten Weges (Zusatzmessung dieser Demo); für Experiment 4 die Überdeckung und Breite der Schranke C bei 20, 100 und 1000
Kalibrierfällen (die 1000 Kalibrierfälle eines Laufs in getrennte Blöcke zerlegt: 50, 10 und 1 Block je Seed); dazu eine Stabilitätsprobe (Rauschstufe 0,5, 100 weitere Zufalls-Seeds 30 bis 129, nur Zusammenfassungen).
Ein Prozess, keine Parallelität; Zufall nur über den ganzzahligen SplitMix64 aus `esc_rng.py`, nie Pythons `hash()`."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from esc_runs import RunSpec, calibration_blocks, cell_result, run_world, summarize  # noqa: E402

DEGS, NOISES, N_CALS = (1, 4, 6), (0.25, 0.5, 0.75), (20, 100, 1000)
KEYS = ("cov_a", "cov_b", "cov_c", "w_a", "w_b", "w_c", "optimism", "optimism_random")


def _r(values, digits=6):
    return [round(float(v), digits) for v in values]


def _block_summary(covs: list, widths: list) -> dict:
    c = np.asarray(covs, dtype=float)
    w = np.asarray(widths, dtype=float)
    finite = w[np.isfinite(w)]
    return {
        "blocks": int(len(c)),
        "coverage": {"mean": round(float(c.mean()), 4), "p05": round(float(np.percentile(c, 5)), 3), "p95": round(float(np.percentile(c, 95)), 3),
                     "min": round(float(c.min()), 3), "max": round(float(c.max()), 3), "share_below_target": round(float(np.mean(c < 90.0)), 4)},
        "width": {"median": round(float(np.median(finite)), 3) if len(finite) else None, "min": round(float(finite.min()), 3) if len(finite) else None,
                  "max": round(float(finite.max()), 3) if len(finite) else None, "share_infinite": round(float(np.mean(~np.isfinite(w))), 4)},
    }


def measure_cell(deg: int, noise: float, seeds) -> dict:
    rows, blocks = [], {n: ([], []) for n in N_CALS}
    for seed in seeds:
        run = run_world(RunSpec(deg=deg, noise=noise, seed=int(seed)))
        rows.append(cell_result(run))
        for n in N_CALS:
            b = calibration_blocks(run, n)
            blocks[n][0].extend(b["coverage"])
            blocks[n][1].extend(b["width"])
    cell = {"deg": deg, "noise": noise, "seeds": [int(s) for s in seeds]}
    for k in KEYS:
        cell[k] = _r(r[k] for r in rows)
    cell["summary"] = {k: summarize(r[k] for r in rows) for k in KEYS}
    cell["blocks"] = {str(n): _block_summary(*blocks[n]) for n in N_CALS}
    return cell


def measure_stability(deg: int, noise: float, seeds) -> dict:
    """Stabilitätsprobe: dieselbe Zelle mit weiteren Zufalls-Seeds (nur Zusammenfassungen)."""
    rows = [cell_result(run_world(RunSpec(deg=deg, noise=noise, seed=int(s)))) for s in seeds]
    return {"deg": deg, "noise": noise, "seeds": [int(s) for s in seeds], "summary": {k: summarize(r[k] for r in rows) for k in KEYS}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "esc_bands.json"))
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    t0 = time.time()
    cells = []
    for deg in DEGS:
        for noise in NOISES:
            cells.append(measure_cell(deg, noise, seeds))
            print(f"Grad {deg}, Rauschstufe {noise}: fertig ({time.time() - t0:.0f} s)", flush=True)
    stability = []
    for deg in DEGS:
        stability.append(measure_stability(deg, 0.5, tuple(range(30, 130))))
        print(f"Stabilitätsprobe Grad {deg}: fertig ({time.time() - t0:.0f} s)", flush=True)
    spec = RunSpec()
    meta = {"seeds": list(seeds), "n_train": spec.n_train, "n_cal": spec.n_cal, "n_test": spec.n_test, "alpha": spec.alpha, "p": spec.p, "h": spec.h, "n_cal_blocks": list(N_CALS),
            "units": "Überdeckung, Breite und Optimismus in Prozent; Breite und Optimismus in Prozent der mittleren vorhergesagten Kosten des gewählten Weges"}
    Path(args.out).write_text(json.dumps({"meta": meta, "cells": cells, "stability": stability}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"geschrieben: {args.out}")


if __name__ == "__main__":
    main()
