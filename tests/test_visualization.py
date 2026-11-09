"""Abbildungen und Tabellenzeilen: jede Figur baut sich für jede Zelle der Auswahl, alle Achsen sind gesperrt, die Zeilen der Experimente haben die erwartete Form."""

import pytest

import esc_constants as C
import esc_evaluation as ev
import esc_visualization as V
from esc_runs import RunSpec, cell_result, run_world

GRID = [(d, e) for d in C.DEG_CHOICES for e in C.NOISE_CHOICES]


def _locked(fig):
    axes = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
    return bool(axes) and all(a.get("fixedrange") is True for a in axes)


@pytest.mark.parametrize("deg,noise", GRID)
def test_figures_build_for_every_cell_and_are_locked(deg, noise):
    rows = ev.bound_rows(deg, noise)
    assert [r["key"] for r in rows] == ["a", "b", "c"]
    assert _locked(V.build_bounds_figure(rows))
    assert _locked(V.build_optimism_figure(ev.optimism_rows(noise)))
    blocks = ev.block_rows(deg, noise)
    assert [r["n"] for r in blocks] == list(C.N_CAL_BLOCKS) and _locked(V.build_blocks_figure(blocks))


def test_hand_figure_for_every_slider_step_and_the_infinite_case():
    for i in range(10):
        h = ev.hand_numbers(round(C.LEVEL_MIN + i * C.LEVEL_STEP, 6))
        fig = V.build_hand_figure(h)
        assert _locked(fig)
        texts = [a.text for a in fig.layout.annotations]
        assert any("unendlich" in t for t in texts) == h["infinite"]


def test_live_diamonds_and_case_figures_for_a_small_world():
    run = run_world(RunSpec(n_train=200, n_cal=100, n_test=200))
    now = cell_result(run)
    fig = V.build_bounds_figure(ev.bound_rows(4, 0.5), now)
    assert len(fig.data) == 4 and _locked(fig)                      # zwei Balken, zwei Rauten-Spuren
    cn = ev.case_numbers(run, 1)
    chat = run.Chat_test[0]
    assert _locked(V.build_grid_figure(run.grid, chat, cn["w"], float(chat.min()), float(chat.max())))
    case = V.build_case_figure(cn)
    assert _locked(case) and list(case.data[0].y) == [cn["pred"], cn["real"], cn["c"], cn["a"], cn["b"]]


def test_case_numbers_cover_flags_follow_the_bounds():
    run = run_world(RunSpec(n_train=200, n_cal=100, n_test=200))
    for case in (1, 2, 50, 200):
        cn = ev.case_numbers(run, case)
        assert cn["covered_c"] == (cn["real"] <= cn["c"]) and cn["covered_a"] == (cn["real"] <= cn["a"]) and cn["covered_b"] == (cn["real"] <= cn["b"])
        assert cn["pred"] == pytest.approx(float(run.Chat_test[case - 1] @ cn["w"]))
    assert ev.uncovered_cases(run, 200) == [i + 1 for i in range(200) if run.real[i] > run.bounds.c[i]]
    assert ev.uncovered_cases(run, 5) == [c for c in ev.uncovered_cases(run, 200) if c <= 5]


def test_optimism_and_block_rows_match_the_band_file():
    rows = ev.optimism_rows(0.5)
    assert [r["deg"] for r in rows] == list(C.DEG_CHOICES) and rows[2]["chosen"]["median"] > rows[1]["chosen"]["median"] > rows[0]["chosen"]["median"]
    blocks = ev.block_rows(4, 0.5)
    assert [r["blocks"] for r in blocks] == [1500, 300, 30] and abs(blocks[0]["formula"] - 100 * 19 / 21) < 1e-9
