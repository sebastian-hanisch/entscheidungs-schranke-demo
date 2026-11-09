"""Konforme Schranken für die Kosten der gewählten Entscheidung (keine Streamlit-Abhängigkeit, nur numpy).

Aufgabe: Ein Modell sagt Bogenkosten ĉ vorher, ein Löser wählt daraus den Weg w(ĉ). Gesucht ist eine obere Schranke U, die die tatsächlichen Kosten c·w(ĉ) dieses Weges
mit Wahrscheinlichkeit mindestens `level` überdeckt: P(c·w(ĉ) <= U) >= level.

Drei Verfahren (Messlogik nach der Vormessung `entscheidungsorientiert-planung/vormessung/eo_v10_konform.py`, hier neu geschrieben):

    A  Summe der Kantenschranken:  U = ĉ·w + Summe über die Bögen des Weges von q_e,  q_e = Konformquantil der Bogenfehler (c_e - ĉ_e) zum Niveau `level`.
    B  wie A mit Bonferroni:        je Bogen das Niveau 1 - (1 - level) / m  (m = Zahl der Bögen des Weges, im 5 x 5 Gitter 8).
    C  konform auf den Wegkosten:   U = ĉ·w + q,  q = Konformquantil des Scores s = c·w(ĉ) - ĉ·w(ĉ) (tatsächliche minus vorhergesagte Kosten des gewählten Weges).

Split-Konform: Das Quantil kommt aus n Kalibrierfällen, die weder zum Training noch zum Test gehören. Mit dem Rang k = ceil((n + 1) * level) ist q der k-kleinste Score;
ist k > n, gibt es keine Garantie und q = unendlich. Die Garantie ist marginal und gilt im Mittel über die Kalibrierungen (Austauschbarkeit der Kalibrier- und Testfälle):
P(s_test <= q) >= k / (n + 1) >= level. Für eine feste Kalibrierung kann die Überdeckung darunter liegen (siehe Experiment 4). Keine Aussage je Merkmalswert.

Einordnung (nur Abstract-Ebene, kein Nachbau): Decision-Focused Uncertainty Quantification (arXiv 2410.01767, in der zweiten Fassung Utility-Directed Conformal Prediction; ein einziges Papier) bildet
konforme Mengen unter Berücksichtigung des Entscheidungsverlusts bei erhaltener Abdeckung. Die Idee „Summenfehler statt Einzelfehler“ steht schon in `forecast-inventory-demo`.
"""

import math
from dataclasses import dataclass

import numpy as np


def conformal_rank(n: int, level: float) -> int:
    """Rang k = ceil((n + 1) * level). Das Produkt wird vor dem Aufrunden auf 9 Stellen gerundet, damit ein Gleitkommarest (90,00000000000001) nicht zum Rang 91 führt."""
    if not 0.0 < level < 1.0:
        raise ValueError("das Niveau muss zwischen 0 und 1 liegen (ausschließlich)")
    if n < 0:
        raise ValueError("die Zahl der Kalibrierfälle darf nicht negativ sein")
    return math.ceil(round((n + 1) * level, 9))


def conformal_quantile(scores, level: float) -> float:
    """Der k-kleinste Score (k = conformal_rank), unendlich bei k > n (zu wenig Kalibrierfälle für das Niveau). Gleichstände ändern den Wert nicht."""
    s = np.sort(np.asarray(scores, dtype=float).ravel())
    k = conformal_rank(len(s), level)
    return math.inf if k > len(s) else float(s[k - 1])


def guaranteed_coverage(n: int, level: float) -> float:
    """Mittlere Überdeckung über Kalibrierungen bei stetigen Scores: k / (n + 1) (genau, wenn die Scores keine Gleichstände haben); 1,0 bei k > n (Schranke unendlich)."""
    k = conformal_rank(n, level)
    return 1.0 if k > n else k / (n + 1)


def path_scores(W: np.ndarray, C: np.ndarray, Chat: np.ndarray) -> np.ndarray:
    """Score je Fall: tatsächliche minus vorhergesagte Kosten des gewählten Weges, c·w - ĉ·w."""
    return np.sum((np.asarray(C, float) - np.asarray(Chat, float)) * W, axis=1)


@dataclass(frozen=True)
class Calibration:
    """Das Ergebnis der Kalibrierung: Quantile der drei Verfahren (je ein Wert, bei A und B je Bogen)."""

    level: float
    n: int
    q_edge: np.ndarray        # Verfahren A: Quantil je Bogen zum Niveau `level`
    q_edge_bonf: np.ndarray   # Verfahren B: Quantil je Bogen zum Niveau 1 - (1 - level) / m
    q_path: float             # Verfahren C: Quantil des Weg-Scores


def calibrate(solver, C_cal: np.ndarray, Chat_cal: np.ndarray, level: float, arcs_per_path: int) -> Calibration:
    """Kalibrierung auf (C_cal, Chat_cal): der Löser wählt je Kalibrierfall den Weg unter ĉ, daraus der Weg-Score."""
    C_cal, Chat_cal = np.asarray(C_cal, float), np.asarray(Chat_cal, float)
    W_cal, _ = solver.solve(Chat_cal)
    edge_scores = C_cal - Chat_cal
    level_bonf = 1.0 - (1.0 - level) / arcs_per_path
    q_edge = np.array([conformal_quantile(edge_scores[:, e], level) for e in range(edge_scores.shape[1])])
    q_edge_b = np.array([conformal_quantile(edge_scores[:, e], level_bonf) for e in range(edge_scores.shape[1])])
    return Calibration(level, len(C_cal), q_edge, q_edge_b, conformal_quantile(path_scores(W_cal, C_cal, Chat_cal), level))


@dataclass(frozen=True)
class Bounds:
    """Gewählte Wege, ihre Vorhersagekosten und die drei oberen Schranken für neue Fälle."""

    W: np.ndarray
    pred: np.ndarray    # vorhergesagte Kosten des gewählten Weges, ĉ·w
    a: np.ndarray       # Summe der Kantenschranken
    b: np.ndarray       # Summe mit Bonferroni
    c: np.ndarray       # konform auf den Wegkosten


def upper_bounds(solver, cal: Calibration, Chat: np.ndarray) -> Bounds:
    """Die drei Schranken für neue Fälle mit den Vorhersagen Chat."""
    W, _ = solver.solve(Chat)
    pred = np.sum(np.asarray(Chat, float) * W, axis=1)
    return Bounds(W, pred, pred + _dot(W, cal.q_edge), pred + _dot(W, cal.q_edge_bonf), pred + cal.q_path)


def _dot(W: np.ndarray, q: np.ndarray) -> np.ndarray:
    """W @ q, aber mit unendlichen Quantilen sauber: ein Bogen mit q = unendlich im Weg macht die Summe unendlich, ein Bogen außerhalb des Weges trägt nichts bei (0 * unendlich bleibt 0)."""
    out = np.zeros(len(W))
    for e in range(W.shape[1]):
        used = W[:, e] > 0
        if used.any():
            out[used] += q[e]
    return out


def coverage(real: np.ndarray, bound: np.ndarray) -> float:
    """Anteil der Fälle mit tatsächlichen Kosten <= Schranke (in [0, 1])."""
    return float(np.mean(np.asarray(real) <= np.asarray(bound)))


def width_percent(bound: np.ndarray, pred: np.ndarray) -> float:
    """Mittlere Breite (Schranke minus Vorhersage) in Prozent der mittleren Vorhersagekosten; unendlich bei einer unendlichen Schranke."""
    return float(100.0 * np.mean(np.asarray(bound) - np.asarray(pred)) / np.mean(pred))


def optimism_percent(real: np.ndarray, pred: np.ndarray) -> float:
    """Optimismus der Vorhersage: (mittlere tatsächliche minus mittlere vorhergesagte Kosten) in Prozent der mittleren vorhergesagten Kosten; positiv heißt zu optimistisch."""
    return float(100.0 * (np.mean(real) - np.mean(pred)) / np.mean(pred))
