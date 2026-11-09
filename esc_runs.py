"""Lauf einer Welt (keine Streamlit-Abhängigkeit, nur numpy): Daten ziehen, KQ anpassen, kalibrieren, die drei Schranken auf den Testfällen messen.

Ein Lauf (`run_world`) zieht Trainings-, Kalibrier- und Testdaten zu einem Zufalls-Seed, passt KQ an, wählt den Weg je Fall mit dem Löser und misst auf den Testfällen
Überdeckung und Breite der drei Schranken sowie den Optimismus der Vorhersage des gewählten Weges. `run_seeds` wiederholt das über feste Seeds, `summarize` gibt Median, Minimum
und Maximum (Median über Seeds, nie ein einzelner Seed als Befund). `calibration_blocks` zerlegt die Kalibrierdaten in Blöcke von n Fällen (Experiment 4).

Als Kontrolle des Optimismus wird zusätzlich ein zufällig gewählter Weg (gleichverteilt über die 70 Wege, fester Zufallsstrom) mit derselben Vorhersage bewertet: Die Auswahl durch den
Löser, nicht die Fehlspezifikation allein, soll sichtbar werden. Das ist eine Zusatzmessung dieser Demo, nicht Teil der Vormessung.
"""

from dataclasses import dataclass, replace

import numpy as np

from esc_conformal import (
    Bounds,
    Calibration,
    calibrate,
    conformal_quantile,
    coverage,
    optimism_percent,
    path_scores,
    upper_bounds,
    width_percent,
)
from esc_grid import Grid
from esc_rng import SplitMix64, derive_seed
from esc_training import fit_least_squares, predict
from esc_world import TAG_RANDOM, Dataset, WorldSpec, make_dataset


@dataclass(frozen=True)
class RunSpec:
    """Alles, was einen Lauf festlegt."""

    deg: int = 4
    noise: float = 0.5
    seed: int = 0
    n_train: int = 1000
    n_cal: int = 1000
    n_test: int = 5000
    p: int = 5
    h: int = 5
    alpha: float = 0.1

    @property
    def world(self) -> WorldSpec:
        return WorldSpec(p=self.p, deg=self.deg, noise=self.noise)

    @property
    def level(self) -> float:
        return 1.0 - self.alpha


@dataclass
class WorldRun:
    spec: RunSpec
    grid: Grid
    data: Dataset
    coef: np.ndarray
    Chat_cal: np.ndarray
    Chat_test: np.ndarray
    cal: Calibration
    bounds: Bounds
    real: np.ndarray            # tatsächliche Kosten des gewählten Weges je Testfall
    pred: np.ndarray            # vorhergesagte Kosten des gewählten Weges je Testfall (= bounds.pred)
    real_random: np.ndarray     # tatsächliche Kosten eines zufällig gewählten Weges (Kontrolle)
    pred_random: np.ndarray     # vorhergesagte Kosten desselben Weges


def run_world(spec: RunSpec) -> WorldRun:
    grid = Grid(spec.h)
    data = make_dataset(spec.world, grid.d, spec.n_train, spec.n_cal, spec.n_test, spec.seed)
    coef = fit_least_squares(data.X_train, data.C_train)
    Chat_cal, Chat_test = predict(coef, data.X_cal), predict(coef, data.X_test)
    cal = calibrate(grid, data.C_cal, Chat_cal, spec.level, grid.steps)
    bounds = upper_bounds(grid, cal, Chat_test)
    real = np.sum(data.C_test * bounds.W, axis=1)
    paths = grid.paths()
    idx = np.minimum((SplitMix64(derive_seed(spec.seed, TAG_RANDOM)).uniform(spec.n_test) * len(paths)).astype(int), len(paths) - 1)
    W_rand = paths[idx]
    return WorldRun(spec, grid, data, coef, Chat_cal, Chat_test, cal, bounds, real, bounds.pred,
                    np.sum(data.C_test * W_rand, axis=1), np.sum(Chat_test * W_rand, axis=1))


def cell_result(run: WorldRun) -> dict:
    """Die Zahlen eines Laufs in Prozent: Überdeckung und Breite der drei Schranken, Optimismus des gewählten und eines zufälligen Weges."""
    b = run.bounds
    return {
        "seed": run.spec.seed,
        "cov_a": 100.0 * coverage(run.real, b.a), "cov_b": 100.0 * coverage(run.real, b.b), "cov_c": 100.0 * coverage(run.real, b.c),
        "w_a": width_percent(b.a, b.pred), "w_b": width_percent(b.b, b.pred), "w_c": width_percent(b.c, b.pred),
        "optimism": optimism_percent(run.real, run.pred),
        "optimism_random": optimism_percent(run.real_random, run.pred_random),
    }


def run_seeds(spec: RunSpec, seeds) -> list[dict]:
    """Läuft über feste Zufalls-Seeds; je Seed ein Wörterbuch aus `cell_result`."""
    return [cell_result(run_world(replace(spec, seed=int(s)))) for s in seeds]


def calibration_blocks(run: WorldRun, n_cal: int) -> dict:
    """Experiment 4: die Kalibrierdaten in aufeinanderfolgende, getrennte Blöcke von je `n_cal` Fällen zerlegen (Rest verworfen); je Block das Quantil des Weg-Scores (Verfahren C)
    und daraus Überdeckung und Breite auf den Testfällen. Rückgabe: Listen je Block (Überdeckung und Breite in Prozent; Breite unendlich, wenn der Rang n + 1 überschreitet)."""
    W_cal, _ = run.grid.solve(run.Chat_cal)
    s_cal = path_scores(W_cal, run.data.C_cal, run.Chat_cal)
    s_test = np.sort(run.real - run.pred)
    mean_pred = float(np.mean(run.pred))
    covs, widths = [], []
    for start in range(0, len(s_cal) - n_cal + 1, n_cal):
        q = conformal_quantile(s_cal[start:start + n_cal], run.spec.level)
        covs.append(100.0 * float(np.searchsorted(s_test, q, side="right")) / len(s_test))
        widths.append(100.0 * q / mean_pred)
    return {"coverage": covs, "width": widths}


def summarize(values) -> dict:
    """Median, Minimum, Maximum einer Liste."""
    a = np.asarray(list(values), dtype=float)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}
