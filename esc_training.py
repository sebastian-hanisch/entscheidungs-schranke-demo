"""Vorhersagemodell: Kleinste Quadrate (KQ) für Bogenkosten (keine Streamlit-Abhängigkeit, nur numpy).

Herkunft: der KQ-Teil von `spo-kuerzester-weg-demo/skw_training.py` (Stück 3 der Reihe), ohne das SPO+-Training: diese Demo prüft nur die Unsicherheit der Entscheidung
eines gegebenen Modells, nicht dessen Training (offene Messlücke: ein auf SPO+ trainiertes Modell ist nicht gemessen).

Modell: lineares Modell mit Achsenabschnitt, ch = [1, x] @ coef, coef hat (p + 1) x d Einträge (5 Merkmale, 40 Bögen: 240 Parameter).
"""

import numpy as np


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(X), 1)), np.asarray(X, float)])


def fit_least_squares(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Kleinste Quadrate: coef = argmin Summe (c - [1, x] coef)^2, über `numpy.linalg.lstsq` (kleinste Norm bei Rangmangel)."""
    coef, *_ = np.linalg.lstsq(add_intercept(X), np.asarray(C, float), rcond=None)
    return coef


def predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return add_intercept(X) @ coef
