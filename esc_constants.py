"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Das Gitter und die Welt ---------------------------------------------------------------------------------------------------------------
GRID_H = 5                       # 5 x 5 Knoten: 40 Bögen, 70 Wege, je 8 Bögen
N_FEATURES = 5                   # Merkmale je Fall
N_TRAIN, N_CAL, N_TEST = 1000, 1000, 5000   # Trainings-, Kalibrier- und Testfälle je Welt
ALPHA = 0.1                      # Fehlerniveau; Ziel-Überdeckung 1 - ALPHA = 0,9
TARGET_PERCENT = 90.0
BAND_SEEDS = tuple(range(30))    # Zufalls-Seeds der vorgerechneten Bänder (esc_bands.json)
N_CAL_BLOCKS = (20, 100, 1000)   # Zahl der Kalibrierfälle in Experiment 4

DEG_CHOICES = (1, 4, 6)          # Grad der Wahrheit: 1 = linear
NOISE_CHOICES = (0.25, 0.5, 0.75)   # Rauschstufe e
DEFAULT_DEG, DEFAULT_NOISE, DEFAULT_SEED = 4, 0.5, 0
SEED_MAX = 999999
CASE_MIN, CASE_MAX, DEFAULT_CASE = 1, 100, 1    # Nummer des gezeigten Testfalls (1 bis 100 von 5000)

# --- Das Handbeispiel (fünf Kalibrierwerte) ---------------------------------------------------------------------------------------------------
HAND_SCORES = (-0.5, 0.2, 0.9, 1.4, 2.0)   # Score = tatsächliche minus vorhergesagte Kosten des gewählten Weges
HAND_PRED = 10.0                 # vorhergesagte Kosten des gewählten Weges eines neuen Falls im Handbeispiel
LEVEL_MIN, LEVEL_MAX, LEVEL_STEP, DEFAULT_LEVEL = 0.5, 0.95, 0.05, 0.8

# --- Anzeige ------------------------------------------------------------------------------------------------------------------------------
COLOR_SUM = "#4c78a8"            # A: Summe der Kantenschranken
COLOR_BONF = "#b8860b"           # B: mit Bonferroni
COLOR_PATH = "#2F6B65"           # C: konform auf den Wegkosten
COLOR_PRED = "#7f7f7f"           # vorhergesagte Kosten
COLOR_REAL = "#d62728"           # tatsächliche Kosten
COLOR_BEST = "#222222"
EDGE_COLORSCALE = "Viridis"

PRESETS = {
    "Grad 4, Rauschstufe 0,5 (Voreinstellung)": {"deg": 4, "noise": 0.5, "seed": 0},
    "Lineare Wahrheit (Grad 1)": {"deg": 1, "noise": 0.5, "seed": 0},
    "Starke Fehlspezifikation (Grad 6)": {"deg": 6, "noise": 0.5, "seed": 0},
    "Wenig Rauschen (Grad 4)": {"deg": 4, "noise": 0.25, "seed": 0},
}
PRESET_HELP = {
    "Grad 4, Rauschstufe 0,5 (Voreinstellung)": "Grad 4, Rauschstufe 0,5, Zufalls-Seed 0: das lineare Modell passt nicht zur Wahrheit, die Schranke auf den Wegkosten ist deutlich schmaler als die Summe der Kantenschranken.",
    "Lineare Wahrheit (Grad 1)": "Grad 1, Rauschstufe 0,5: das lineare Modell kann die Wahrheit exakt darstellen; die Vorhersage des gewählten Weges ist fast unverzerrt, die Summe der Kantenschranken überdeckt in allen 30 Zufalls-Seeds der Messung 100 %.",
    "Starke Fehlspezifikation (Grad 6)": "Grad 6, Rauschstufe 0,5: starke Abweichung der Wahrheit vom linearen Modell; die Vorhersage des gewählten Weges ist am deutlichsten zu optimistisch, die Breite der Schranke am größten.",
    "Wenig Rauschen (Grad 4)": "Grad 4, Rauschstufe 0,25: weniger Rauschen, dieselbe Fehlspezifikation; der Fehler des Modells bleibt, die Schranken werden enger.",
}
