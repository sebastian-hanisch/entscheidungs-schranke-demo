"""Kennzahlen, Verdikt-Logik und Tabellen für die App (keine Streamlit-Abhängigkeit): Handbeispiel, Lauf einer Welt, vorgerechnete Bänder.

Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`, `tests/test_verdicts.py` prüft Schwellen und jede Zahl im Text an handgesetzten Werten.
Zahlen erscheinen mit Dezimalkomma (`de`)."""

import math
from dataclasses import dataclass


import esc_bands as B
import esc_constants as C
from esc_conformal import conformal_quantile, conformal_rank, guaranteed_coverage
from esc_runs import RunSpec


def _shown(value: float, digits: int = 1) -> float:
    """Der Wert so, wie er mit `digits` Stellen angezeigt wird (Vergleiche stimmen mit dem überein, was die Leser sehen)."""
    return float(f"{float(value):.{digits}f}")


def de(value: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma; negative Nullen (-0,0) erscheinen als 0,0."""
    text = f"{float(value):.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace(".", ",")


@dataclass(frozen=True)
class Settings:
    deg: int = C.DEFAULT_DEG
    noise: float = C.DEFAULT_NOISE
    seed: int = C.DEFAULT_SEED
    case: int = C.DEFAULT_CASE
    level: float = C.DEFAULT_LEVEL

    def run_spec(self) -> RunSpec:
        return RunSpec(deg=self.deg, noise=self.noise, seed=self.seed, n_train=C.N_TRAIN, n_cal=C.N_CAL, n_test=C.N_TEST, p=C.N_FEATURES, h=C.GRID_H, alpha=C.ALPHA)


# --- Experiment 1: der Score von Hand ---------------------------------------------------------------------------------------------------------

def hand_numbers(level: float, scores=C.HAND_SCORES, pred: float = C.HAND_PRED) -> dict:
    """Handbeispiel: Rang, Schranke und garantierte mittlere Überdeckung bei n Kalibrierwerten; `bound` = vorhergesagte Kosten eines neuen Falls plus Schranke (unendlich bei zu kleinem n)."""
    n = len(scores)
    q = conformal_quantile(scores, level)
    k = conformal_rank(n, level)
    return {
        "n": n, "level": level, "k": k, "sorted": tuple(sorted(float(s) for s in scores)), "q": q, "infinite": math.isinf(q),
        "pred": pred, "bound": pred + q, "guarantee": guaranteed_coverage(n, level), "max_level": n / (n + 1),
    }


def verdict_hand(h: dict) -> tuple[str, str]:
    """Verdikt des Handbeispiels aus den Zahlen von `hand_numbers`."""
    head = f"Rang k = ⌈({h['n']} + 1) · {de(h['level'], 2)}⌉ = {h['k']}"
    if h["infinite"]:
        return "warning", (
            f"{head} liegt über n = {h['n']}: keine Garantie bei n = {h['n']}, die Schranke ist unendlich. Mit {h['n']} Kalibrierwerten ist höchstens das Niveau "
            f"{h['n']} / {h['n'] + 1} = {de(100 * h['max_level'])} % erreichbar."
        )
    return "success", (
        f"{head}: der {h['k']}. kleinste Score, {de(h['q'])}, ist die Schranke. Für einen neuen Fall mit vorhergesagten Kosten {de(h['pred'])} heißt das: tatsächliche Kosten höchstens {de(h['bound'])}. "
        f"Sind die {h['n']} Werte und der neue Fall austauschbar, ist die Überdeckung im Mittel über Kalibrierungen mindestens {h['k']} / {h['n'] + 1} = {de(100 * h['guarantee'])} %."
    )


# --- Experiment 2: drei Schranken ----------------------------------------------------------------------------------------------------------------

METHODS = (("a", "Summe der Kantenschranken"), ("b", "Summe mit Bonferroni"), ("c", "Konform auf den Wegkosten"))


def bound_rows(deg: int, noise: float) -> list[dict]:
    """Zeilen der Tabelle für eine Zelle: je Verfahren Überdeckung und Breite (Median, Minimum, Maximum über die Zufalls-Seeds, Prozent)."""
    s = B.cell(deg, noise)["summary"]
    return [{"Verfahren": name, "key": key, "coverage": s[f"cov_{key}"], "width": s[f"w_{key}"]} for key, name in METHODS]


def verdict_bounds(cov_a: float, cov_b: float, cov_c: float, w_a: float, w_b: float, w_c: float, seeds: int) -> tuple[str, str]:
    """Verdikt über Überdeckung und Breite der drei Schranken (Medianwerte über `seeds` Zufalls-Seeds; Überdeckung in Prozent, Breite in Prozent der Vorhersagekosten)."""
    target = C.TARGET_PERCENT
    scope = f"Median über {seeds} Zufalls-Seeds, Ziel {de(target, 0)} %"
    head = (
        f"Die Schranke auf den Wegkosten überdeckt {de(cov_c)} % bei einer Breite von {de(w_c)} %, die Summe der Kantenschranken {de(cov_a)} % bei {de(w_a)} % "
        f"(mit Bonferroni {de(cov_b)} % bei {de(w_b)} %; {scope}; Breite in Prozent der Vorhersagekosten)."
    )
    if not (target - 1.0 <= _shown(cov_c) <= target + 1.0):
        return "warning", head + " Die Überdeckung der Wegkosten-Schranke liegt mehr als einen Prozentpunkt vom Ziel entfernt."
    if _shown(cov_a) < target:
        return "warning", head + " Die Summe der Kantenschranken verfehlt das Ziel."
    if _shown(w_c) < _shown(w_a):
        return "success", head + f" Beide treffen oder übertreffen das Ziel; die Summe ist dafür {de(w_a - w_c)} Prozentpunkte breiter."
    return "info", head + " In dieser Zelle ist die Wegkosten-Schranke nicht schmaler als die Summe der Kantenschranken."


# --- Experiment 3: Optimismus der Auswahl ---------------------------------------------------------------------------------------------------------

def optimism_rows(noise: float) -> list[dict]:
    """Je Grad: Optimismus des gewählten Weges und des zufällig gewählten Weges (Median, Minimum, Maximum über die Zufalls-Seeds, Prozent der vorhergesagten Kosten)."""
    rows = []
    for deg in C.DEG_CHOICES:
        s = B.cell(deg, noise)["summary"]
        rows.append({"deg": deg, "chosen": s["optimism"], "random": s["optimism_random"]})
    return rows


def verdict_optimism(chosen_median: float, chosen_min: float, chosen_max: float, random_median: float, seeds: int, deg: int, noise: float) -> tuple[str, str]:
    """Verdikt über den Optimismus (Prozent der vorhergesagten Kosten) des gewählten Weges, mit dem zufällig gewählten Weg als Kontrolle."""
    scope = f"Grad {deg}, Rauschstufe {de(noise, 2)}, Median über {seeds} Zufalls-Seeds, Spanne {de(chosen_min)} bis {de(chosen_max)} %"
    control = f"Bei einem zufällig gewählten Weg derselben Vorhersage liegt der Median bei {de(random_median)} %."
    if _shown(chosen_median) > 0:
        return "warning", f"Die Vorhersage des gewählten Weges ist im Median {de(chosen_median)} % zu optimistisch ({scope}). {control}"
    if _shown(chosen_median) == 0:
        return "info", f"Die Vorhersage des gewählten Weges ist im Median nicht optimistisch (0,0 %; {scope}). {control}"
    return "info", f"Die Vorhersage des gewählten Weges ist im Median {de(-chosen_median)} % zu pessimistisch ({scope}). {control}"


# --- Experiment 4: Kalibrierfälle ---------------------------------------------------------------------------------------------------------------

def block_rows(deg: int, noise: float) -> list[dict]:
    """Je Zahl der Kalibrierfälle: Überdeckung und Breite der Wegkosten-Schranke über alle Kalibrierungen der 30 Zufalls-Seeds, dazu die Rang-Formel k / (n + 1)."""
    rows = []
    for n in C.N_CAL_BLOCKS:
        b = B.cell(deg, noise)["blocks"][str(n)]
        rows.append({"n": n, "blocks": b["blocks"], "coverage": b["coverage"], "width": b["width"], "formula": 100 * guaranteed_coverage(n, 1.0 - C.ALPHA)})
    return rows


def verdict_blocks(small: dict, large: dict) -> tuple[str, str]:
    """Verdikt über die kleinste und die größte Zahl der Kalibrierfälle (Zeilen aus `block_rows`)."""
    return "info", (
        f"Mit {small['n']} Kalibrierfällen liegt die Überdeckung im Mittel über {small['blocks']} Kalibrierungen bei {de(small['coverage']['mean'])} % (Ziel {de(C.TARGET_PERCENT, 0)} %), "
        f"streut aber von {de(small['coverage']['p05'])} bis {de(small['coverage']['p95'])} % (5. bis 95. Perzentil) und liegt in {de(100 * small['coverage']['share_below_target'], 0)} % der Kalibrierungen unter dem Ziel; "
        f"mit {large['n']} Kalibrierfällen sind es im Mittel {de(large['coverage']['mean'])} % bei {de(large['coverage']['p05'])} bis {de(large['coverage']['p95'])} %. "
        f"Breite: Median {de(small['width']['median'])} (Spanne {de(small['width']['min'])} bis {de(small['width']['max'])}) gegen {de(large['width']['median'])} "
        f"({de(large['width']['min'])} bis {de(large['width']['max'])}) % der Vorhersagekosten."
    )


# --- Ein Testfall ------------------------------------------------------------------------------------------------------------------------------

def case_numbers(run, case: int) -> dict:
    """Zahlen eines Testfalls (Nummer ab 1): gewählter Weg, vorhergesagte und tatsächliche Kosten, die drei Schranken und ob sie überdecken."""
    i = case - 1
    b = run.bounds
    real = float(run.real[i])
    return {
        "w": b.W[i], "pred": float(b.pred[i]), "real": real,
        "a": float(b.a[i]), "b": float(b.b[i]), "c": float(b.c[i]),
        "covered_a": real <= float(b.a[i]), "covered_b": real <= float(b.b[i]), "covered_c": real <= float(b.c[i]),
    }


def uncovered_cases(run, limit: int = C.CASE_MAX) -> list[int]:
    """Nummern (ab 1) der ersten `limit` Testfälle, in denen die tatsächlichen Kosten über der Wegkosten-Schranke liegen."""
    return [i + 1 for i in range(min(limit, len(run.real))) if run.real[i] > run.bounds.c[i]]


def next_case(cases: list[int], current: int) -> int | None:
    """Nächste Nummer der Liste nach `current` (am Ende von vorn); None bei leerer Liste."""
    if not cases:
        return None
    later = [c for c in cases if c > current]
    return later[0] if later else cases[0]


def verdict_case(case: int, pred: float, real: float, bound_c: float, bound_a: float) -> tuple[str, str]:
    """Verdikt eines einzelnen Testfalls (Kosten in Kosteneinheiten)."""
    head = f"Testfall {case}: vorhergesagt {de(pred, 2)}, tatsächlich {de(real, 2)}; Schranke auf den Wegkosten {de(bound_c, 2)}, Summe der Kantenschranken {de(bound_a, 2)}."
    if real > bound_c:
        return "warning", head + " Die tatsächlichen Kosten liegen über der Wegkosten-Schranke; bei dem Ziel von 90 % ist das im Mittel über Kalibrierungen in höchstens jedem zehnten Fall zu erwarten."
    return "success", head + " Die tatsächlichen Kosten liegen nicht über der Wegkosten-Schranke."


# --- Diese Welt (eine Instanz) und der Verlauf über den Grad -----------------------------------------------------------------------------------

def verdict_instance(now: dict, seed: int, cov_c_min: float, cov_c_max: float, seeds: int) -> tuple[str, str]:
    """Verdikt einer einzelnen Welt (ein Zufalls-Seed, Zahlen aus `esc_runs.cell_result`), eingeordnet an der Spanne der Wegkosten-Überdeckung über `seeds` Zufalls-Seeds des Bandes."""
    return "info", (
        f"In dieser Welt (Zufalls-Seed {seed}) überdeckt die Schranke auf den Wegkosten {de(now['cov_c'])} % bei einer Breite von {de(now['w_c'])} %, die Summe der Kantenschranken "
        f"{de(now['cov_a'])} % bei {de(now['w_a'])} %. Eine einzelne Kalibrierung trifft das Ziel nicht genau: über {seeds} Zufalls-Seeds lag die Überdeckung der Wegkosten-Schranke zwischen "
        f"{de(cov_c_min)} und {de(cov_c_max)} %."
    )


def verdict_optimism_trend(rows: list[dict], noise: float, seeds: int) -> tuple[str, str]:
    """Verdikt über alle Grade (Zeilen aus `optimism_rows`): wächst der Optimismus des gewählten Weges mit dem Grad, und wie groß ist die Kontrolle mit zufälligem Weg?"""
    meds = [r["chosen"]["median"] for r in rows]
    parts = ", ".join(f"Grad {r['deg']}: {de(r['chosen']['median'])} %" for r in rows)
    control = max(abs(r["random"]["median"]) for r in rows)
    tail = f"Bei einem zufällig gewählten Weg liegt der Median in allen Graden im Betrag höchstens bei {de(control)} % (Rauschstufe {de(noise, 2)}, Median über {seeds} Zufalls-Seeds)."
    if all(_shown(a) < _shown(b) for a, b in zip(meds, meds[1:])):
        return "warning", f"Der Optimismus des gewählten Weges wächst mit dem Grad der Wahrheit: {parts}. {tail}"
    return "info", f"Der Optimismus des gewählten Weges wächst nicht in jedem Schritt mit dem Grad: {parts}. {tail}"
