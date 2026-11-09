"""Presets und Permalink: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellung, Bänder zu jedem Preset und jeder Zelle vorhanden."""

import pytest

import esc_bands as B
import esc_constants as C
import esc_evaluation as ev
import esc_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_allowed_choices_and_inside_the_bounds():
    for name, p in C.PRESETS.items():
        assert p["deg"] in C.DEG_CHOICES and p["noise"] in C.NOISE_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name


def test_presets_differ_from_each_other():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_default_preset_is_the_first_and_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    d = ev.Settings()
    assert C.PRESETS[first] == {"deg": d.deg, "noise": d.noise, "seed": d.seed}
    assert {k: spec.default for k, spec in P.SETTING_SPECS.items()} == {
        "deg_slider": d.deg, "noise_slider": d.noise, "seed_input": d.seed, "case_slider": d.case, "level_slider": d.level,
    }


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)


def test_every_preset_and_every_cell_of_the_choice_grid_has_a_band_with_all_seeds():
    for p in C.PRESETS.values():
        assert B.cell(p["deg"], p["noise"])["cov_c"]
    for deg in C.DEG_CHOICES:
        for noise in C.NOISE_CHOICES:
            c = B.cell(deg, noise)
            assert c["seeds"] == list(C.BAND_SEEDS) and len(c["cov_c"]) == len(C.BAND_SEEDS)
            assert set(c["blocks"]) == {str(n) for n in C.N_CAL_BLOCKS}


def test_hand_example_defaults_are_a_valid_slider_state():
    assert C.LEVEL_MIN <= C.DEFAULT_LEVEL <= C.LEVEL_MAX and C.CASE_MIN <= C.DEFAULT_CASE <= C.CASE_MAX


@pytest.mark.parametrize("deg,noise", [(d, e) for d in C.DEG_CHOICES for e in C.NOISE_CHOICES])
def test_settings_run_spec_matches_the_constants(deg, noise):
    spec = ev.Settings(deg=deg, noise=noise).run_spec()
    assert (spec.p, spec.h, spec.n_train, spec.n_cal, spec.n_test, spec.alpha) == (C.N_FEATURES, C.GRID_H, C.N_TRAIN, C.N_CAL, C.N_TEST, C.ALPHA)
    assert (spec.deg, spec.noise) == (deg, noise) and abs(spec.level - C.TARGET_PERCENT / 100) < 1e-12
