"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an seine Grenzen
und prüft, dass kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key= können zufällig
identischen Inhalt rendern und kollidieren) und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max). Dazu: jedes Preset zeigt die erwarteten Verdikte."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import esc_constants as C
import esc_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return [e.value for e in list(at.success) + list(at.info) + list(at.warning)]


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_select_slider_and_number_input_at_min_and_max_does_not_raise():
    for state_key, spec in P.SETTING_SPECS.items():
        for edge, value in (("min", spec.lo), ("max", spec.hi)):
            at = _fresh_app()
            found = False
            for art in ("slider", "select_slider", "number_input"):
                try:
                    widget = getattr(at, art)(key=state_key)
                except KeyError:
                    continue
                widget.set_value(value).run(timeout=TIMEOUT)
                found = True
                break
            assert found, state_key
            assert not at.exception, (state_key, edge, [str(e) for e in at.exception])


def test_permalink_snaps_values_to_the_allowed_choices():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["deg"] = "3"
    at.query_params["noise"] = "0.3"
    at.query_params["seed"] = "-4"
    at.query_params["case"] = "500"
    at.query_params["level"] = "0.83"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    s = at.session_state
    assert (s["deg_slider"], s["noise_slider"], s["seed_input"], s["case_slider"], s["level_slider"]) == (4, 0.25, 0, C.CASE_MAX, 0.85)


def test_permalink_with_garbage_values_falls_back_to_the_defaults():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["deg"] = "abc"
    at.query_params["level"] = "nan"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["deg_slider"] == C.DEFAULT_DEG and at.session_state["level_slider"] == C.DEFAULT_LEVEL


def test_permalink_roundtrip_writes_the_settings_to_the_address():
    at = _fresh_app()
    at.select_slider(key="deg_slider").set_value(6).run(timeout=TIMEOUT)
    assert dict(at.query_params) == {"deg": "6", "noise": "0.5", "seed": "0", "case": "1", "level": "0.8"}


@pytest.mark.parametrize("name,fragments", [
    ("Grad 4, Rauschstufe 0,5 (Voreinstellung)", ("Grad 4, Rauschstufe 0,50", "zu optimistisch")),
    ("Lineare Wahrheit (Grad 1)", ("Grad 1, Rauschstufe 0,50", "zu optimistisch")),
    ("Starke Fehlspezifikation (Grad 6)", ("Grad 6, Rauschstufe 0,50", "zu optimistisch")),
    ("Wenig Rauschen (Grad 4)", ("Grad 4, Rauschstufe 0,25", "zu optimistisch")),
])
def test_every_preset_shows_the_expected_core_question_verdicts(name, fragments):
    at = _fresh_app()
    at.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    texts = _texts(at)
    assert any("überdeckt" in t and "Die Schranke auf den Wegkosten" in t and "Median über 30 Zufalls-Seeds" in t for t in texts), (name, texts)
    assert any(all(f in t for f in fragments) for t in texts), (name, texts)
    assert any(t.startswith("Der Optimismus des gewählten Weges wächst mit dem Grad") for t in texts)


def test_all_four_experiments_the_boundaries_and_the_formulas_are_there():
    at = _fresh_app()
    headings = [s.value for s in at.subheader]
    for i in range(1, 5):
        assert any(h.startswith(f"🔬 Experiment {i}:") for h in headings), i
    assert "🚧 Wo die Annahmen enden" in headings
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)
    assert any(e.label.startswith("So funktioniert") for e in at.expander)


def test_hand_example_shows_the_infinite_bound_at_level_090_and_a_finite_one_at_080():
    at = _fresh_app()
    assert any("der 5. kleinste Score, 2,0" in t for t in _texts(at))              # Standardniveau 0,8
    at.slider(key="level_slider").set_value(0.9).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("keine Garantie bei n = 5" in w.value for w in at.warning)


def test_next_case_button_jumps_to_a_case_above_the_path_bound():
    from esc_evaluation import Settings, uncovered_cases
    from esc_runs import run_world

    at = _fresh_app()
    cases = uncovered_cases(run_world(Settings().run_spec()))
    assert cases, "Standardwelt: mindestens ein Testfall über der Wegkosten-Schranke unter den ersten 100"
    at.button(key="next_case").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["case_slider"] == cases[0]
    assert any(t.startswith(f"Testfall {cases[0]}:") and "über der Wegkosten-Schranke" in t for t in [w.value for w in at.warning])


def test_pdf_button_creates_a_download():
    at = _fresh_app()
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["pdf_made"][1][:5] == b"%PDF-"
