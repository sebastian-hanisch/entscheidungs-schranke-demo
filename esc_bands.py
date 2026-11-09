"""Vorgerechnete Bänder (keine Streamlit-Abhängigkeit): `esc_bands.json` (eigene Messung, `tools/measure_bands.py`, Zufalls-Seeds 0 bis 29)."""

import json
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def own() -> dict:
    return json.loads((HERE / "esc_bands.json").read_text(encoding="utf-8"))


def cell(deg: int, noise: float) -> dict:
    """Eigene Messung einer Zelle (Grad, Rauschstufe): Listen je Seed, Zusammenfassungen und die Blöcke der Kalibrierfälle."""
    for c in own()["cells"]:
        if (c["deg"], c["noise"]) == (deg, noise):
            return c
    raise KeyError((deg, noise))
