"""Claims: jede Zahl der README steht hier als wörtliche Tabellenzeile, Band, Zählung oder Ungleichung (DEMO-PLAYBOOK §11a).

Zahlen-Tabelle der README -> Test:

| README | Test |
|---|---|
| Tabellen Überdeckung, Breite, Optimismus, Kalibrierfälle, Stabilitätsprobe, Plan gegen Messung | `test_readme_contains_every_generated_table_line` (wörtlich aus der JSON-Datei erzeugt) |
| Wegkosten-Schranke trifft das Ziel (90,0 bis 90,2 %; Läufe 87,4 bis 92,4 %) | `test_path_bound_hits_the_target_in_every_cell` |
| Summe überdeckt mehr als nötig (93,4 bis 100,0 %; 100,0 % bei Grad 1; nie unter 91,4 %) | `test_sum_of_edge_bounds_overcovers` |
| Wegkosten-Schranke schmaler (2,6 bis 42,0 Punkte; 264 von 270; die 6 Ausnahmen) | `test_path_bound_is_narrower_than_the_sum` |
| Bonferroni breiter in 270 von 270, 99,7 bis 100,0 % | `test_bonferroni_is_wider_and_overcovers` |
| Optimismus (0,1 bis 7,4 %, wächst mit Grad und Rauschstufe, 258 von 270 Läufen positiv) | `test_optimism_is_positive_and_grows_with_degree_and_noise` |
| Kontrolle mit zufälligem Weg (höchstens 0,13 %, Läufe −1,5 bis 2,0 %) | `test_random_path_control_has_no_optimism` |
| Kalibrierfälle: Mittel gegen Rang-Formel, Anteil unter 90 %, 5. bis 95. Perzentil | `test_mean_coverage_equals_the_rank_formula`, `test_single_calibrations_fall_below_the_target`, `test_beta_share_matches_for_small_n` |
| Breite und Kalibrierfälle (Median, Spanne) | `test_width_median_flattens_and_spread_shrinks_with_more_calibration_cases` |
| Stabilitätsprobe (0,2 / 1,4 / 0,3 Punkte) | `test_stability_probe_confirms_the_medians` |
| Korrekturen gegenüber dem Plan (Abweichungen 0,2 / 12,8 / 1,7) | `test_plan_numbers_versus_this_measurement` |
| Handbeispiel (Ränge, Schranken, Obergrenzen) | `test_handbeispiel_numbers_of_the_readme` (und `test_conformal.py`, `test_verdicts.py`) |
| Preset-Hilfetexte (Grad 1: 100 % in allen 30 Seeds; Grad 6 breiteste Schranke; weniger Rauschen enger) | `test_preset_help_texts_hold` |
| Verweise auf die Schwester-Demos (88,5 %, 95,0 %, 31. von 30) | `test_references_match_the_sibling_readmes` (übersprungen, wenn die Ordner fehlen) |
| Messung dieser Demo (Seeds, Größen) | `test_band_file_describes_the_documented_setup`, `test_live_run_reproduces_the_bands` |
| Löser-, Rang-, Quantil- und Schranken-Prüfungen der README (Scratch-Läufe) | nicht im Repo: Scratch-Läufe mit 600399, 40000 und 200 Prüfungen; im Repo die kleine Fassung in `test_oracle_entscheidungs_schranke.py` |
| Testzahl | `test_readme_states_the_number_of_tests` |

Gemessene Zahlen hängen von der Plattform ab (Normalverteilung aus `log` und `cos`): gegen die Bänder wird in Ungleichungen und mit Toleranz geprüft, exakt nur Handbeispiel, Rangformel und Hand-Instanzen (`test_conformal.py`).
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import esc_bands as B
import esc_constants as C
import readme_tables as T
from esc_conformal import guaranteed_coverage
from esc_evaluation import de, hand_numbers
from esc_runs import RunSpec, run_seeds

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
CELLS = B.own()["cells"]
GRID = [(d, e) for d in C.DEG_CHOICES for e in C.NOISE_CHOICES]


def _med(deg, noise, key):
    return B.cell(deg, noise)["summary"][key]["median"]


def _runs(key, deg=None):
    return [v for c in CELLS if deg is None or c["deg"] == deg for v in c[key]]


def test_readme_contains_every_generated_table_line():
    blocks = [T.coverage_table(), T.width_table(), T.optimism_table(), T.blocks_table(), T.stability_table(), T.plan_table()]
    missing = [line for lines in blocks for line in lines if line not in README]
    assert not missing, missing[:3]


def test_band_file_describes_the_documented_setup():
    meta = B.own()["meta"]
    assert meta["seeds"] == list(C.BAND_SEEDS) == list(range(30)) and meta["n_train"] == C.N_TRAIN and meta["n_cal"] == C.N_CAL and meta["n_test"] == C.N_TEST
    assert meta["alpha"] == C.ALPHA and meta["n_cal_blocks"] == list(C.N_CAL_BLOCKS)
    assert len(CELLS) == 9 and all(len(c["cov_c"]) == 30 for c in CELLS)
    assert all(e["seeds"] == list(range(30, 130)) and e["noise"] == 0.5 for e in B.own()["stability"])
    for text in ("Zufalls-Seeds 0 bis 29", "1000 Trainings-, 1000 Kalibrier- und 5000 Testfälle", "Zufalls-Seeds 30 bis 129"):
        assert text in README


def test_live_run_reproduces_the_bands():
    """Zwei Seeds einer Zelle neu gerechnet: dieselben Zahlen wie in der JSON-Datei (Toleranz für die letzte Stelle der Normalverteilung: 0,05 Punkte)."""
    cell = B.cell(4, 0.5)
    rows = run_seeds(RunSpec(deg=4, noise=0.5), [0, 1])
    for i, r in enumerate(rows):
        for key in ("cov_a", "cov_b", "cov_c", "w_a", "w_b", "w_c", "optimism", "optimism_random"):
            assert r[key] == pytest.approx(cell[key][i], abs=0.05), (i, key)


def test_path_bound_hits_the_target_in_every_cell():
    assert all(89.9 <= _med(d, e, "cov_c") <= 90.3 for d, e in GRID)
    assert de(min(_med(d, e, "cov_c") for d, e in GRID)) == "90,0" and de(max(_med(d, e, "cov_c") for d, e in GRID)) == "90,2"
    runs = _runs("cov_c")
    assert len(runs) == 270 and de(min(runs)) == "87,4" and de(max(runs)) == "92,4"
    assert "zwischen 90,0 und 90,2 %" in README and "zwischen 87,4 und 92,4 %" in README


def test_sum_of_edge_bounds_overcovers():
    meds = [_med(d, e, "cov_a") for d, e in GRID]
    assert de(min(meds)) == "93,4" and de(max(meds)) == "100,0" and all(m > 90.0 for m in meds)
    assert min(_runs("cov_a", 1)) > 99.95 and len(_runs("cov_a", 1)) == 90                   # Grad 1: 100,0 % in allen 90 Läufen
    assert de(min(_runs("cov_a", 4) + _runs("cov_a", 6))) == "91,4" and len(_runs("cov_a", 4) + _runs("cov_a", 6)) == 180
    assert min(_runs("cov_a")) > 90.0                                                          # Ziel in keinem der 270 Läufe verfehlt
    assert sum(v > 99.995 for v in _runs("cov_a", 1)) == 80 and round(min(_runs("cov_a", 1)), 2) == 99.98     # 80 von 90 Läufen bei 100,00 %, der Rest bei 99,98 %
    assert "zwischen 93,4 und 100,0 %" in README and "keinem der 180 Läufe unter 91,4 %" in README and "in allen 90 Läufen bei mindestens 99,98 % (in 80 bei 100,0 %)" in README


def test_path_bound_is_narrower_than_the_sum():
    diffs = {(d, e): _med(d, e, "w_a") - _med(d, e, "w_c") for d, e in GRID}
    assert all(v > 0 for v in diffs.values())
    assert de(min(diffs.values())) == "2,6" and min(diffs, key=diffs.get) == (6, 0.25) and de(max(diffs.values())) == "42,0" and max(diffs, key=diffs.get) == (1, 0.75)
    assert [de(diffs[(d, 0.5)]) for d in C.DEG_CHOICES] == ["28,1", "22,2", "12,4"]
    assert (de(_med(6, 0.25, "w_c")), de(_med(6, 0.25, "w_a"))) == ("51,8", "54,3") and (de(_med(1, 0.75, "w_c")), de(_med(1, 0.75, "w_a"))) == ("21,0", "63,0")
    narrower = sum(1 for c in CELLS for a, b in zip(c["w_c"], c["w_a"]) if a < b)
    assert narrower == 264
    exceptions = [(c["deg"], c["noise"], s) for c in CELLS for s, a, b in zip(c["seeds"], c["w_c"], c["w_a"]) if not a < b]
    assert len(exceptions) == 6 and {(d, e) for d, e, _ in exceptions} == {(6, 0.25)}
    for text in ("um 2,6 ", "bis 42,0 Prozentpunkte", "264 von 270 Läufen", "(6 von 30 Seeds)", "28,1 Punkte bei Grad 1, 22,2 bei Grad 4 und 12,4 bei Grad 6"):
        assert text in README


def test_bonferroni_is_wider_and_overcovers():
    assert all(b > a for c in CELLS for a, b in zip(c["w_a"], c["w_b"]))                    # in allen 270 Läufen breiter
    assert de(min(_runs("cov_b"))) == "99,7" and max(_runs("cov_b")) <= 100.0
    assert (de(_med(6, 0.5, "w_b")), de(_med(6, 0.5, "w_a"))) == ("225,6", "65,3")
    assert "in allen 270 Läufen breiter" in README and "99,7 bis 100,0 %" in README


def test_optimism_is_positive_and_grows_with_degree_and_noise():
    meds = {(d, e): _med(d, e, "optimism") for d, e in GRID}
    assert all(m > 0 for m in meds.values()) and de(min(meds.values())) == "0,1" and de(max(meds.values())) == "7,4"
    for e in C.NOISE_CHOICES:
        assert meds[(1, e)] < meds[(4, e)] < meds[(6, e)]
    for d in C.DEG_CHOICES:
        assert meds[(d, 0.25)] < meds[(d, 0.5)] < meds[(d, 0.75)]
    assert [de(meds[(d, 0.5)]) for d in C.DEG_CHOICES] == ["0,4", "2,3", "7,0"] and [de(meds[(4, e)]) for e in C.NOISE_CHOICES] == ["2,1", "2,3", "2,6"]
    positive = {(c["deg"], c["noise"]): sum(1 for v in c["optimism"] if v > 0) for c in CELLS}
    assert sum(positive.values()) == 258 and positive[(1, 0.25)] == 21 and positive[(1, 0.5)] == 27
    assert all(v == 30 for k, v in positive.items() if k[0] != 1) and positive[(1, 0.75)] == 30
    for text in ("0,1 bis 7,4 %", "258 von 270 Läufen", "9 bei Rauschstufe 0,25, 3 bei 0,5", "0,4 / 2,3 / 7,0 %", "2,1 / 2,3 / 2,6 %"):
        assert text in README


def test_random_path_control_has_no_optimism():
    meds = [abs(_med(d, e, "optimism_random")) for d, e in GRID]
    assert round(max(meds), 2) == 0.13
    assert all(_med(d, e, "optimism") > _med(d, e, "optimism_random") for d, e in GRID)       # in allen 9 Zellen liegt der Optimismus des gewählten Weges über dem des zufälligen
    runs = _runs("optimism_random")
    assert de(min(runs)) == "-1,5" and de(max(runs)) == "2,0"
    assert "höchstens 0,13 %" in README and "zwischen −1,5 und 2,0 %" in README


def test_mean_coverage_equals_the_rank_formula():
    for deg in C.DEG_CHOICES:
        for n in C.N_CAL_BLOCKS:
            b = B.cell(deg, 0.5)["blocks"][str(n)]
            assert abs(b["coverage"]["mean"] - 100 * guaranteed_coverage(n, 0.9)) < 0.12, (deg, n)
    assert [round(100 * guaranteed_coverage(n, 0.9), 2) for n in C.N_CAL_BLOCKS] == [90.48, 90.10, 90.01]      # 19/21, 91/101, 901/1001
    assert "höchstens 0,11 Punkte" in README and "89,99 bis 90,05 %" in README


def test_single_calibrations_fall_below_the_target():
    small = [B.cell(d, 0.5)["blocks"]["20"]["coverage"] for d in C.DEG_CHOICES]
    large = [B.cell(d, 0.5)["blocks"]["1000"]["coverage"] for d in C.DEG_CHOICES]
    assert [round(100 * c["share_below_target"]) for c in small] == [39, 37, 38]
    assert [round(100 * c["share_below_target"]) for c in large] == [43, 47, 53]
    assert de(min(c["p05"] for c in small)) == "77,9" and de(max(c["p95"] for c in small)) == "98,3"
    assert de(min(c["p05"] for c in large)) == "87,9" and de(max(c["p05"] for c in large)) == "88,4"
    assert all(c["min"] < 90.0 for c in small + large)
    assert B.cell(4, 0.5)["blocks"]["20"]["coverage"]["min"] == 57.0 and "das Minimum liegt bei 57,0 %" in README
    for text in ("37 bis 39 %", "zwischen 77,9 und 98,3 %", "43 bis 53 %", "87,9 bis 88,4 %", "77,9 und 98,1 %"):
        assert text in README


def test_beta_share_matches_for_small_n():
    stats = pytest.importorskip("scipy.stats")
    for n, k in ((20, 19), (100, 91)):
        theory = float(stats.beta.cdf(0.9, k, n + 1 - k))
        for deg in C.DEG_CHOICES:
            assert abs(B.cell(deg, 0.5)["blocks"][str(n)]["coverage"]["share_below_target"] - theory) < 0.05, (deg, n)
    assert round(100 * float(stats.beta.cdf(0.9, 19, 2))) == 39 and "Beta-Formel: 39 %" in README


def test_width_median_flattens_and_spread_shrinks_with_more_calibration_cases():
    w = {d: [B.cell(d, 0.5)["blocks"][str(n)]["width"] for n in C.N_CAL_BLOCKS] for d in C.DEG_CHOICES}
    drop_20_100 = [w[d][0]["median"] - w[d][1]["median"] for d in C.DEG_CHOICES]
    change_100_1000 = [abs(w[d][2]["median"] - w[d][1]["median"]) for d in C.DEG_CHOICES]
    assert [de(v) for v in drop_20_100] == ["0,9", "2,7", "6,6"] and max(change_100_1000) < 0.75
    assert w[6][2]["median"] > w[6][1]["median"] and (de(w[6][1]["median"]), de(w[6][2]["median"])) == ("52,2", "52,9")          # Grad 6 steigt von 100 auf 1000
    for d in C.DEG_CHOICES:
        spreads = [x["max"] - x["min"] for x in w[d]]
        assert spreads[0] > spreads[1] > spreads[2], d
    assert (de(w[4][0]["min"]), de(w[4][0]["max"]), de(w[4][1]["min"]), de(w[4][1]["max"]), de(w[4][2]["min"]), de(w[4][2]["max"])) == ("1,6", "75,2", "13,3", "41,0", "19,7", "29,4")
    for text in ("um 0,9 (Grad 1), 2,7 (Grad 4) und 6,6 Punkte (Grad 6)", "höchstens 0,7 Punkte (bei Grad 6 steigt er leicht)", "1,6 bis 75,2 % bei n = 20, 13,3 bis 41,0 % bei n = 100, 19,7 bis 29,4 % bei n = 1000", "0,9 bis 6,6 Punkte"):
        assert text in README


def test_stability_probe_confirms_the_medians():
    for e in B.own()["stability"]:
        deg, s = e["deg"], e["summary"]
        cell = B.cell(deg, 0.5)["summary"]
        assert abs(s["cov_c"]["median"] - cell["cov_c"]["median"]) <= 0.25
        assert abs(s["w_c"]["median"] - cell["w_c"]["median"]) <= 1.5
        assert abs(s["optimism"]["median"] - cell["optimism"]["median"]) <= 0.35
    deltas = [abs(e["summary"]["w_c"]["median"] - _med(e["deg"], 0.5, "w_c")) for e in B.own()["stability"]]
    assert 1.4 < max(deltas) <= 1.5                                                           # Grad 6: 52,93 gegen 51,48 (Differenz der Rohwerte 1,46; der Unterschied der gerundeten Werte 52,9 und 51,5 wäre 1,4)
    assert "höchstens 0,2 Punkte ab, ihre Breite um höchstens 1,5 Punkte" in README and "höchstens 0,3 Punkte" in README


def test_plan_numbers_versus_this_measurement():
    for deg in C.DEG_CHOICES:
        s = B.cell(deg, 0.5)["summary"]
        for key in ("a", "b", "c"):
            pc, pw = T.PLAN[deg][key]
            assert abs(s[f"cov_{key}"]["median"] - pc) <= 0.25 and abs(s[f"w_{key}"]["median"] - pw) <= 12.9, (deg, key)
        assert abs(s["optimism"]["median"] - T.PLAN[deg]["optimism"]) <= 0.2
    gaps_c = [abs(B.cell(d, 0.5)["summary"]["w_c"]["median"] - T.PLAN[d]["c"][1]) for d in C.DEG_CHOICES]
    assert round(max(gaps_c), 1) == 1.7
    assert "höchstens 0,2 Punkte ab, die Breiten um höchstens 12,8 Punkte" in README and "um höchstens 1,7 Punkte" in README


def test_handbeispiel_numbers_of_the_readme():
    assert [hand_numbers(lv)["k"] for lv in (0.5, 0.8, 0.9)] == [3, 5, 6]
    assert [hand_numbers(lv)["q"] for lv in (0.5, 0.8)] == [0.9, 2.0] and hand_numbers(0.9)["infinite"]
    assert [hand_numbers(lv)["bound"] for lv in (0.5, 0.8)] == [10.9, 12.0]
    for text in ("Schranke 0,9", "Schranke 2,0", "**unendlich**", "10,9 beziehungsweise 12,0", "$5/6$"):
        assert text in README


def test_preset_help_texts_hold():
    """Hilfetexte der Presets sind Behauptungen: Grad 1 mit Rauschstufe 0,5: Summe überdeckt in allen 30 Seeds 100 %; Grad 6 hat die breiteste Wegkosten-Schranke; weniger Rauschen macht alle drei Schranken enger."""
    assert min(B.cell(1, 0.5)["cov_a"]) > 99.95 and "in allen 30 Zufalls-Seeds der Messung 100 %" in C.PRESET_HELP["Lineare Wahrheit (Grad 1)"]
    assert _med(6, 0.5, "w_c") > _med(4, 0.5, "w_c") > _med(1, 0.5, "w_c") and _med(6, 0.5, "optimism") > _med(4, 0.5, "optimism") > _med(1, 0.5, "optimism")
    for key in ("w_a", "w_b", "w_c"):
        assert _med(4, 0.25, key) < _med(4, 0.5, key), key
    assert _med(4, 0.5, "w_c") < _med(4, 0.5, "w_a") / 1.5                                     # Grad 4: Wegkosten-Schranke deutlich schmaler (24,2 gegen 46,4 %)
    assert "deutlich schmaler" in C.PRESET_HELP["Grad 4, Rauschstufe 0,5 (Voreinstellung)"]


def test_references_match_the_sibling_readmes():
    inv = ROOT.parent / "forecast-inventory-demo" / "README.md"
    interval = ROOT.parent / "forecast-interval-demo" / "README.md"
    if not (inv.exists() and interval.exists()):
        pytest.skip("die Schwester-Demos liegen nicht neben diesem Repo (CI): Verweistexte wurden beim Bau gegen die README geprüft")
    a, b = inv.read_text(encoding="utf-8"), interval.read_text(encoding="utf-8")
    assert "88,5 %" in a and "konforme Regel liegt bei α 95,0 %" in a and "Faustregel" in a and "Summenfehler" in a
    assert "31. von 30" in b
    for text in ("α = 88,5 % bei Ziel 95 %", "konform 95,0 %", "31. von 30 Werten", "zweiseitigen 95-%-Intervall"):
        assert text in README


def test_readme_states_the_number_of_tests():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, stdin=subprocess.DEVNULL)
    m = re.search(r"(\d+) tests? collected", out.stdout)
    assert m, out.stdout[-300:]
    assert f"**{m.group(1)} Tests**" in README
