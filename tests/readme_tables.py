"""Erzeugt die Zahlentabellen der README aus den vorgerechneten Bändern (`esc_bands.json`) und der Rang-Formel.

`tests/test_claims.py` prüft, dass jede dieser Zeilen wörtlich in der README steht: die Zahlen der README sind damit nie von Hand abgeschrieben, und ändert sich eine Messung,
schlägt der Test an. Ausgabe zum Einfügen: `python tests/readme_tables.py`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import esc_bands as B  # noqa: E402
import esc_constants as C  # noqa: E402
from esc_conformal import guaranteed_coverage  # noqa: E402
from esc_evaluation import de  # noqa: E402

# Zahlen der Vormessung (Rohausgabe `entscheidungsorientiert-planung/vormessung/out_eo_v10_konform.txt`, Zufallsquelle der Vormessung, 30 Seeds, Rauschstufe 0,5):
# je Grad (Überdeckung % Median, Breite % Median) für Summe, Bonferroni, Wegkosten; dazu der Optimismus.
PLAN = {
    1: {"a": (100.0, 41.6), "b": (100.0, 54.9), "c": (90.0, 13.9), "optimism": 0.40},
    4: {"a": (98.4, 46.0), "b": (100.0, 111.0), "c": (90.1, 23.0), "optimism": 2.13},
    6: {"a": (95.1, 64.1), "b": (99.9, 212.8), "c": (90.1, 51.2), "optimism": 7.15},
}


def _band(s: dict) -> str:
    return f"{de(s['median'])} [{de(s['min'])}; {de(s['max'])}]"


def coverage_table() -> list[str]:
    lines = ["| Grad | Rauschstufe | Summe der Kantenschranken | mit Bonferroni | konform auf den Wegkosten |", "|---|---|---|---|---|"]
    for deg in C.DEG_CHOICES:
        for noise in C.NOISE_CHOICES:
            s = B.cell(deg, noise)["summary"]
            lines.append(f"| {deg} | {de(noise, 2)} | {_band(s['cov_a'])} | {_band(s['cov_b'])} | {_band(s['cov_c'])} |")
    return lines


def width_table() -> list[str]:
    lines = ["| Grad | Rauschstufe | Summe der Kantenschranken | mit Bonferroni | konform auf den Wegkosten |", "|---|---|---|---|---|"]
    for deg in C.DEG_CHOICES:
        for noise in C.NOISE_CHOICES:
            s = B.cell(deg, noise)["summary"]
            lines.append(f"| {deg} | {de(noise, 2)} | {_band(s['w_a'])} | {_band(s['w_b'])} | {_band(s['w_c'])} |")
    return lines


def optimism_table() -> list[str]:
    lines = ["| Grad | Rauschstufe | gewählter Weg | zufällig gewählter Weg (Kontrolle) |", "|---|---|---|---|"]
    for deg in C.DEG_CHOICES:
        for noise in C.NOISE_CHOICES:
            s = B.cell(deg, noise)["summary"]
            lines.append(f"| {deg} | {de(noise, 2)} | {_band(s['optimism'])} | {_band(s['optimism_random'])} |")
    return lines


def blocks_table(noise: float = 0.5) -> list[str]:
    """Experiment 4 (Rauschstufe `noise`): Überdeckung der Wegkosten-Schranke je Zahl der Kalibrierfälle über alle Kalibrierungen der 30 Zufalls-Seeds."""
    stats = _beta()
    lines = ["| Grad | Kalibrierfälle n | Kalibrierungen | Überdeckung Mittel | Rang-Formel k / (n + 1) | 5. bis 95. Perzentil | Anteil unter 90 % | Anteil unter 90 % (Beta-Formel) | Breite Median [Min; Max] |",
             "|---|---|---|---|---|---|---|---|---|"]
    for deg in C.DEG_CHOICES:
        for n in C.N_CAL_BLOCKS:
            b = B.cell(deg, noise)["blocks"][str(n)]
            k = round(guaranteed_coverage(n, 0.9) * (n + 1))
            beta_share = float(stats.beta.cdf(0.9, k, n + 1 - k))
            c, w = b["coverage"], b["width"]
            lines.append(
                f"| {deg} | {n} | {b['blocks']} | {de(c['mean'], 2)} % | {de(100 * guaranteed_coverage(n, 0.9), 2)} % | {de(c['p05'])} bis {de(c['p95'])} % | "
                f"{de(100 * c['share_below_target'], 0)} % | {de(100 * beta_share, 0)} % | {de(w['median'])} [{de(w['min'])}; {de(w['max'])}] % |"
            )
    return lines


def stability_table() -> list[str]:
    lines = ["| Grad | Überdeckung Summe | Überdeckung Wegkosten | Breite Summe | Breite Wegkosten | Optimismus |", "|---|---|---|---|---|---|"]
    for e in B.own()["stability"]:
        s = e["summary"]
        lines.append(f"| {e['deg']} | {_band(s['cov_a'])} | {_band(s['cov_c'])} | {_band(s['w_a'])} | {_band(s['w_c'])} | {_band(s['optimism'])} |")
    return lines


def plan_table() -> list[str]:
    """Plan (Vormessung) gegen diese Messung, Rauschstufe 0,5: Median über 30 Zufalls-Seeds, Überdeckung und Breite je Verfahren, Optimismus."""
    names = {"a": "Summe der Kantenschranken", "b": "mit Bonferroni", "c": "konform auf den Wegkosten"}
    lines = ["| Grad | Verfahren | Vormessung: Überdeckung / Breite | diese Messung: Überdeckung / Breite |", "|---|---|---|---|"]
    for deg in C.DEG_CHOICES:
        s = B.cell(deg, 0.5)["summary"]
        for key, name in names.items():
            pc, pw = PLAN[deg][key]
            lines.append(f"| {deg} | {name} | {de(pc)} % / {de(pw)} % | {de(s['cov_' + key]['median'])} % / {de(s['w_' + key]['median'])} % |")
        lines.append(f"| {deg} | Optimismus des gewählten Weges | {de(PLAN[deg]['optimism'], 2)} % | {de(s['optimism']['median'], 2)} % |")
    return lines


def _beta():
    import scipy.stats as stats

    return stats


if __name__ == "__main__":
    for name, fn in (("Überdeckung", coverage_table), ("Breite", width_table), ("Optimismus", optimism_table), ("Kalibrierfälle", blocks_table), ("Stabilität", stability_table), ("Plan", plan_table)):
        print(f"\n### {name}\n")
        print("\n".join(fn()))
