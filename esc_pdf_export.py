"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile, Klammern und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import esc_bands as B
import esc_constants as C
from esc_evaluation import Settings, bound_rows, de, hand_numbers, optimism_rows, verdict_bounds, verdict_hand, verdict_optimism
from esc_runs import cell_result

_REPLACE = {"→": "->", "–": "-", "—": "-", "ĉ": "c^", "−": "-", "≥": ">=", "·": "*", "×": "x", "σ": "sigma", "€": "EUR", "²": "^2", "ˆ": "^", "⌈": "ceil(", "⌉": ")"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, run) -> bytes:
    """`run`: das `WorldRun` zu den Einstellungen (Live-Lauf einer Instanz)."""
    now = cell_result(run)
    rows = bound_rows(settings.deg, settings.noise)
    cell = B.cell(settings.deg, settings.noise)
    seeds = len(cell["seeds"])
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _s("Unsicherheit mitliefern: eine Schranke für die Kosten der gewählten Entscheidung"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"Welt: 5 x 5 Gitter, 5 Merkmale, Grad {settings.deg}, Rauschstufe {de(settings.noise, 2)}, Zufalls-Seed {settings.seed}; {run.spec.n_train} Trainings-, {run.spec.n_cal} Kalibrier- und "
        f"{run.spec.n_test} Testfälle, Ziel-Überdeckung {de(100 * run.spec.level, 0)} %. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s("Diese Instanz (ein Zufalls-Seed)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for key, name in (("a", "Summe der Kantenschranken"), ("b", "Summe mit Bonferroni"), ("c", "Konform auf den Wegkosten")):
        pdf.multi_cell(0, 5, _s(f"{name}: Überdeckung {de(now['cov_' + key])} %, Breite {de(now['w_' + key])} % der Vorhersagekosten."), new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 5, _s(f"Optimismus der Vorhersage des gewählten Weges: {de(now['optimism'])} % (zufällig gewählter Weg: {de(now['optimism_random'])} %)."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s(f"Median über {seeds} Zufalls-Seeds (vorgerechnet)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for r in rows:
        pdf.multi_cell(0, 5, _s(
            f"{r['Verfahren']}: Überdeckung {de(r['coverage']['median'])} % (Spanne {de(r['coverage']['min'])} bis {de(r['coverage']['max'])} %), "
            f"Breite {de(r['width']['median'])} % (Spanne {de(r['width']['min'])} bis {de(r['width']['max'])} %)"), new_x="LMARGIN", new_y="NEXT")
    cov, wid = {r["key"]: r["coverage"]["median"] for r in rows}, {r["key"]: r["width"]["median"] for r in rows}
    _, text = verdict_bounds(cov["a"], cov["b"], cov["c"], wid["a"], wid["b"], wid["c"], seeds)
    pdf.ln(1)
    pdf.multi_cell(0, 5, _s(text.replace("**", "")), new_x="LMARGIN", new_y="NEXT")
    row = next(r for r in optimism_rows(settings.noise) if r["deg"] == settings.deg)
    _, text = verdict_optimism(row["chosen"]["median"], row["chosen"]["min"], row["chosen"]["max"], row["random"]["median"], seeds, settings.deg, settings.noise)
    pdf.multi_cell(0, 5, _s(text.replace("**", "")), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s("Handbeispiel: fünf Kalibrierwerte"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _, text = verdict_hand(hand_numbers(settings.level))
    pdf.multi_cell(0, 5, _s(f"Scores {'; '.join(de(s) for s in C.HAND_SCORES)}, Niveau {de(settings.level, 2)}. " + text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Unsicherheit mitliefern: konforme Schranken für die Kosten der gewählten Entscheidung im Gitter. Quelle: entscheidungs-schranke-demo, Sebastian Hanisch."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
