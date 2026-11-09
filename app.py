"""Unsicherheit mitliefern: eine Schranke für die Kosten der gewählten Entscheidung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Im 5 x 5 Gitter wählt ein Löser aus vorhergesagten Bogenkosten den kürzesten Weg; eine konforme Schranke direkt auf den tatsächlichen Kosten dieses Weges wird mit der Summe
der Kantenschranken verglichen, dazu der Optimismus der Vorhersage des gewählten Weges.
Aufbau nach DEMO-PLAYBOOK §2. Kern ohne Streamlit: esc_rng, esc_grid, esc_world, esc_training, esc_conformal, esc_runs.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import esc_bands as B
import esc_constants as C
from esc_evaluation import (
    Settings,
    block_rows,
    bound_rows,
    case_numbers,
    de,
    hand_numbers,
    next_case,
    optimism_rows,
    uncovered_cases,
    verdict_blocks,
    verdict_bounds,
    verdict_case,
    verdict_hand,
    verdict_instance,
    verdict_optimism,
    verdict_optimism_trend,
)
from esc_layout import dunkles_design, kopf_css
from esc_pdf_export import build_pdf
from esc_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from esc_runs import cell_result, run_world
from esc_visualization import (
    build_blocks_figure,
    build_bounds_figure,
    build_case_figure,
    build_grid_figure,
    build_hand_figure,
    build_optimism_figure,
)

st.set_page_config(page_title="Unsicherheit mitliefern – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Welt wird gerechnet: Trainings-, Kalibrier- und Testfälle ziehen, Modell anpassen, Wege wählen und kalibrieren …")
def _world_run(deg: int, noise: float, seed: int):
    return run_world(Settings(deg=deg, noise=noise, seed=seed).run_spec())


def _pct(x: float) -> str:
    return f"{de(x)} %"


def _band(s: dict) -> str:
    return f"{de(s['median'])} [{de(s['min'])}; {de(s['max'])}]"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in esc_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🛡️ Unsicherheit mitliefern: eine Schranke für die Kosten der gewählten Entscheidung")
st.markdown(
    """
Wer eine Entscheidung aus einer Prognose ableitet, will wissen, **wie teuer sie höchstens wird**, nicht wie falsch die Prognose höchstens ist. Hier im 5 × 5 Gitter: ein lineares Modell sagt die Kosten der 40 Bögen voraus,
ein Löser wählt daraus den kürzesten Weg, und eine **konforme Schranke** wird direkt auf die **tatsächlichen Kosten des gewählten Weges** gelegt, statt Fehlerschranken je Bogen zu addieren.
Dazu der **Optimismus der Auswahl**: Die Vorhersage genau des Weges, den der Löser wählt, liegt in den gemessenen Welten im Median unter den tatsächlichen Kosten.

Weiter unten: **🔬 vier Experimente**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Schranke", expanded=True):
    st.markdown(
        """
1. **Vorhersagen und wählen:** ein lineares Modell (Kleinste Quadrate) macht aus den Merkmalen einen Kostenvektor $\\hat c$; der Löser wählt den kürzesten Weg $w(\\hat c)$. Er hat vorhergesagte Kosten $\\hat c^\\top w$ und in Wahrheit Kosten $c^\\top w$.
2. **Den Score bilden:** auf **Kalibrierfällen** (weder im Training noch im Test) ist der Score die tatsächlichen minus die vorhergesagten Kosten des gewählten Weges, $s = c^\\top w - \\hat c^\\top w$.
3. **Das Quantil nehmen:** bei $n$ Kalibrierfällen und Niveau $1 - \\alpha$ ist die Schranke der $k$-kleinste Score mit $k = \\lceil (n+1)(1-\\alpha) \\rceil$. Für einen neuen Fall gilt: tatsächliche Kosten höchstens $\\hat c^\\top w + s_{(k)}$. Ist $k > n$, gibt es keine Garantie, und die Schranke ist unendlich.
4. **Die Gegenprobe:** die Summe der Kantenschranken nimmt je Bogen das Quantil des Bogenfehlers $c_e - \\hat c_e$ und addiert sie über die 8 Bögen des Weges (auch mit Bonferroni: Niveau $1 - \\alpha / 8$ je Bogen).
5. **Wie weit die Garantie reicht:** sie gilt im Mittel über Kalibrierungen und über Fälle (Austauschbarkeit von Kalibrier- und Testfällen), nicht für jede einzelne Kalibrierung und nicht je Merkmalswert.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

# --- Seitenleiste: Schnellstart und Kostenwelt -------------------------------------------------------------------------------------------------

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Kostenwelt**")
    deg = st.select_slider("Grad der Wahrheit", options=list(C.DEG_CHOICES), key="deg_slider",
                           help="1 = die Kosten hängen linear von den Merkmalen ab (ein lineares Modell kann sie exakt abbilden); je größer, desto stärker weicht die Wahrheit von jedem linearen Modell ab.")
    noise = st.select_slider("Rauschstufe", options=list(C.NOISE_CHOICES), key="noise_slider", format_func=lambda v: de(v, 2),
                             help="Jeder Bogen kostet das Mittel mal einen unabhängigen Faktor aus [1 − e, 1 + e].")
    seed = st.number_input("Zufalls-Seed der Welt", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Welt generieren", width="stretch", on_click=randomize_seed)
    st.caption(
        f"Fest: {C.N_TRAIN} Trainings-, {C.N_CAL} Kalibrier- und {C.N_TEST} Testfälle, ein lineares Modell (Kleinste Quadrate), Ziel-Überdeckung {de(C.TARGET_PERCENT, 0)} %. "
        "Das Niveau des Handbeispiels (Experiment 1) und der gezeigte Testfall (Experiment 2) haben ihre Regler beim Experiment."
    )

run = _world_run(int(deg), float(noise), int(seed))
now = cell_result(run)
cell = B.cell(int(deg), float(noise))
n_seeds = len(cell["seeds"])
summary = cell["summary"]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis dieser Welt")
st.caption(
    f"Eine Instanz (Zufalls-Seed {int(seed)}): {C.N_TRAIN} Trainings-, {C.N_CAL} Kalibrier- und {C.N_TEST} Testfälle. Überdeckung = Anteil der Testfälle, in denen die tatsächlichen Kosten des gewählten Weges "
    "höchstens die Schranke erreichen; Breite = mittlere Schranke minus Vorhersage, in Prozent der mittleren vorhergesagten Kosten des gewählten Weges."
)
m1, m2, m3, m4, m5 = st.columns(5)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Überdeckung Wegkosten-Schranke", _pct(now["cov_c"]), delta=f"{de(now['cov_c'] - C.TARGET_PERCENT)} Punkte gegen das Ziel", delta_color="off",
          help="Konform auf den Wegkosten, Ziel 90 %. Eine einzelne Kalibrierung weicht vom Ziel ab (Experiment 4).")
m2.metric("Überdeckung Summe der Kantenschranken", _pct(now["cov_a"]), delta=f"{de(now['cov_a'] - C.TARGET_PERCENT)} Punkte gegen das Ziel", delta_color="off",
          help="Je Bogen ein Konformquantil, über die 8 Bögen des Weges addiert; für den gewählten Weg ohne Garantie.")
m3.metric("Breite Wegkosten-Schranke", _pct(now["w_c"]), delta=f"{de(now['w_c'] - now['w_a'])} Punkte gegen die Summe", delta_color="inverse",
          help="Schranke minus Vorhersage in Prozent der Vorhersagekosten; kleiner ist schmaler.")
m4.metric("Breite Summe der Kantenschranken", _pct(now["w_a"]), delta=f"{de(now['w_a'] - now['w_c'])} Punkte gegen die Wegkosten", delta_color="inverse")
m5.metric("Optimismus der Vorhersage", _pct(now["optimism"]), delta=f"{de(now['optimism'] - now['optimism_random'])} Punkte gegen einen zufälligen Weg", delta_color="off",
          help="Tatsächliche minus vorhergesagte Kosten des gewählten Weges in Prozent der vorhergesagten; positiv heißt zu optimistisch.")

settings = Settings(int(deg), float(noise), int(seed), int(st.session_state.get("case_slider", C.DEFAULT_CASE)), float(st.session_state.get("level_slider", C.DEFAULT_LEVEL)))
if st.button("📄 PDF dieser Welt erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, run))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="entscheidungs-schranke.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie teuer wird der gewählte Weg höchstens?")
rows_b = bound_rows(int(deg), float(noise))
med_cov = {r["key"]: r["coverage"]["median"] for r in rows_b}
med_w = {r["key"]: r["width"]["median"] for r in rows_b}
kind, text = verdict_bounds(med_cov["a"], med_cov["b"], med_cov["c"], med_w["a"], med_w["b"], med_w["c"], n_seeds)
getattr(st, kind)(text)
opt = summary["optimism"]
kind, text = verdict_optimism(opt["median"], opt["min"], opt["max"], summary["optimism_random"]["median"], n_seeds, int(deg), float(noise))
getattr(st, kind)(text)
st.caption(
    f"Beide Verdikte gelten für diese Zelle (Grad {int(deg)}, Rauschstufe {de(noise, 2)}) und das lineare Modell: Median über {n_seeds} Zufalls-Seeds, vorgerechnet mit `tools/measure_bands.py`. "
    "Die Wegkosten-Schranke hat eine Garantie (im Mittel über Kalibrierungen), die Summe der Kantenschranken für den gewählten Weg nicht; ihre Überdeckung ist hier gemessen, nicht zugesichert."
)

st.markdown("---")

# --- Experiment 1 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: der Score von Hand")
st.markdown(
    "Fünf Kalibrierfälle, in denen die tatsächlichen Kosten des gewählten Weges um $-0{,}5$; $0{,}2$; $0{,}9$; $1{,}4$ und $2{,}0$ über der Vorhersage lagen (der **Score**). "
    "Wählen Sie das **Niveau**: Die Schranke ist der Score mit dem Rang $k = \\lceil (n + 1) \\cdot \\text{Niveau} \\rceil$, bei $k > n$ gibt es keine."
)
level = st.slider("Niveau", C.LEVEL_MIN, C.LEVEL_MAX, key="level_slider", step=C.LEVEL_STEP, format="%.2f", help="Gewünschte Überdeckung der Schranke; mit 5 Kalibrierwerten ist höchstens 5/6 = 83,3 % erreichbar.")
hn = hand_numbers(float(level))
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rang k", f"{hn['k']} von {hn['n']}", help="k = ⌈(n + 1) · Niveau⌉. Bei k > n fehlen Kalibrierfälle für dieses Niveau.")
k2.metric("Schranke", "∞" if hn["infinite"] else de(hn["q"]), help="Der k-kleinste der fünf Scores; ∞ heißt unendlich (k > n, keine Garantie).")
k3.metric("Garantie (Mittel)", "keine" if hn["infinite"] else f"{de(100 * hn['guarantee'])} %", help="Garantierte Überdeckung k / (n + 1), im Mittel über Kalibrierungen bei austauschbaren Fällen.")
k4.metric("Obergrenze", "∞" if hn["infinite"] else de(hn["bound"]), help=f"Für einen neuen Fall mit vorhergesagten Kosten {de(C.HAND_PRED)}: vorhergesagte Kosten plus Schranke.")
st.plotly_chart(build_hand_figure(hn), width="stretch", key="hand_chart")
kind, text = verdict_hand(hn)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: drei Schranken im Gitter")
st.markdown(
    f"Dieselbe Welt, drei obere Schranken für die tatsächlichen Kosten des gewählten Weges: die **Summe der Kantenschranken**, dieselbe **mit Bonferroni** und die **konforme Schranke direkt auf den Wegkosten**. "
    f"Balken: Median über {n_seeds} Zufalls-Seeds (vorgerechnet) für Grad {int(deg)} und Rauschstufe {de(noise, 2)} aus der Seitenleiste, Fehlerbalken von Minimum bis Maximum; Rauten: die eine Welt oben. Gestrichelt: das Ziel 90 %."
)
st.plotly_chart(build_bounds_figure(rows_b, now), width="stretch", key="bounds_chart")
table2 = pd.DataFrame([
    {"Verfahren": r["Verfahren"], "Überdeckung (Median [Min; Max])": f"{_band(r['coverage'])} %", "Breite (Median [Min; Max])": f"{_band(r['width'])} %",
     "Diese Welt": f"{de(now['cov_' + r['key']])} % bei {de(now['w_' + r['key']])} %"} for r in rows_b
])
st.dataframe(table2, hide_index=True, width="stretch")
kind, text = verdict_instance(now, int(seed), summary["cov_c"]["min"], summary["cov_c"]["max"], n_seeds)
getattr(st, kind)(text)

st.markdown("**Ein Testfall:** das Gitter mit den **vorhergesagten** Kosten (Farbe und Zahl: violett billig, gelb teuer) und dem Weg, den der Löser wählt.")
uncovered = uncovered_cases(run)
cs1, cs2 = st.columns([3, 1])
with cs1:
    case = st.slider("Testfall Nr.", C.CASE_MIN, C.CASE_MAX, key="case_slider", help=f"Einer der ersten {C.CASE_MAX} von {C.N_TEST} Testfällen dieser Welt.")
with cs2:
    st.write("")
    nxt = next_case(uncovered, int(case))
    st.button("➡️ Nächster Fall über der Wegkosten-Schranke", key="next_case", width="stretch", on_click=lambda n: st.session_state.update(case_slider=n), args=(nxt,), disabled=nxt is None,
              help="Springt zum nächsten der ersten 100 Testfälle, in dem die tatsächlichen Kosten über der Wegkosten-Schranke liegen (am Ende von vorn).")
st.caption(f"In {len(uncovered)} der ersten {C.CASE_MAX} Testfälle liegen die tatsächlichen Kosten über der Wegkosten-Schranke.")
cn = case_numbers(run, int(case))
chat = run.Chat_test[int(case) - 1]
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(build_grid_figure(run.grid, chat, cn["w"], float(chat.min()), float(chat.max())), width="stretch", key="grid_case")
with g2:
    st.plotly_chart(build_case_figure(cn), width="stretch", key="case_chart")
kind, text = verdict_case(int(case), cn["pred"], cn["real"], cn["c"], cn["a"])
getattr(st, kind)(text)
st.caption("Ein einzelner Testfall sagt wenig über die Schranke; Überdeckung und Breite stehen oben als Mittelwerte über alle 5000 Testfälle und als Median über Seeds. Das lineare Modell kennt keine Untergrenze: vorhergesagte Bogenkosten können negativ sein (so bei Grad 4 und 6), die tatsächlichen Kosten sind es in diesen Welten nicht.")

st.markdown("---")

# --- Experiment 3 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: der Optimismus der Auswahl")
st.markdown(
    f"Tatsächliche minus vorhergesagte Kosten des gewählten Weges, in Prozent der vorhergesagten, für Grad 1, 4 und 6 bei Rauschstufe {de(noise, 2)} aus der Seitenleiste (Median über {n_seeds} Zufalls-Seeds, Balken von Minimum bis Maximum). "
    "Zur **Kontrolle** dieselbe Vorhersage, bewertet an einem **zufällig** gewählten Weg (gleichverteilt über die 70 Wege); die Zahlen stehen daneben. Die naheliegende Lesart ist der "
    "**Gewinnerfluch**: Der Löser wählt gerade den Weg, der in der Prognose billig aussieht, und dort ist der Fehler der Prognose im Mittel nach unten gerichtet."
)
rows_o = optimism_rows(float(noise))
st.plotly_chart(build_optimism_figure(rows_o), width="stretch", key="optimism_chart")
table3 = pd.DataFrame([
    {"Grad": r["deg"], "Gewählter Weg (Median [Min; Max])": f"{_band(r['chosen'])} %", "Zufällig gewählter Weg (Median [Min; Max])": f"{_band(r['random'])} %"} for r in rows_o
])
st.dataframe(table3, hide_index=True, width="stretch")
kind, text = verdict_optimism_trend(rows_o, float(noise), n_seeds)
getattr(st, kind)(text)
st.caption("Dass es der Gewinnerfluch ist, ist eine Deutung: Die Kontrolle mit zufälligem Weg stützt sie, ein Versuch, der die Auswahl allein variiert (etwa Löser mit Rauschen), ist nicht gemacht.")

st.markdown("---")

# --- Experiment 4 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: wie viele Kalibrierfälle?")
st.markdown(
    f"Die {C.N_CAL} Kalibrierfälle jeder Welt werden in getrennte Blöcke von 20, 100 oder 1000 Fällen zerlegt (50, 10 und 1 Block je Welt, {n_seeds} Zufalls-Seeds, Grad {int(deg)}, Rauschstufe {de(noise, 2)}). "
    "Jeder Block ergibt eine Schranke; gemessen wird ihre Überdeckung auf den 5000 Testfällen. Links: Mittel und Balken vom 5. bis 95. Perzentil der Überdeckung, gestrichelt das Ziel; rechts: Median und Spanne der Breite in Prozent der Vorhersagekosten. **Marginal** heißt: im Mittel über Kalibrierungen. Die einzelne Kalibrierung kann darunter liegen."
)
rows_n = block_rows(int(deg), float(noise))
st.plotly_chart(build_blocks_figure(rows_n), width="stretch", key="blocks_chart")
table4 = pd.DataFrame([
    {"Kalibrierfälle n": r["n"], "Kalibrierungen": r["blocks"], "Überdeckung Mittel": f"{de(r['coverage']['mean'], 2)} %", "Rang-Formel k / (n + 1)": f"{de(r['formula'], 2)} %",
     "5. bis 95. Perzentil": f"{de(r['coverage']['p05'])} bis {de(r['coverage']['p95'])} %", "Minimum bis Maximum": f"{de(r['coverage']['min'])} bis {de(r['coverage']['max'])} %",
     "Anteil unter 90 %": f"{de(100 * r['coverage']['share_below_target'], 0)} %", "Breite (Median [Min; Max])": f"{_band(r['width'])} %"} for r in rows_n
])
st.dataframe(table4, hide_index=True, width="stretch")
kind, text = verdict_blocks(rows_n[0], rows_n[-1])
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
_all_a = [c["summary"]["cov_a"]["median"] for c in B.own()["cells"]]
_all_c = [c["summary"]["cov_c"]["median"] for c in B.own()["cells"]]
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist |
|---|---|
| **Kalibrier- und Testfälle sind austauschbar** (gleiche Verteilung, unabhängig vom Training) | Die Garantie gilt nur dann. Ein Verteilungswechsel zwischen Kalibrierung und Einsatz ist in dieser Demo nicht gemessen; die Zeitreihen-Demo [forecast-interval-demo](https://github.com/sebastian-hanisch/forecast-interval-demo) misst ihn für Prognoseintervalle. |
| **Genug Kalibrierfälle für das Niveau** | Bei $k > n$ ist die Schranke unendlich (Experiment 1: 5 Werte, Niveau 0,9). Auch bei $k \\le n$ schwankt die Überdeckung einer einzelnen Kalibrierung (Experiment 4). |
| **Die Garantie ist marginal** | Sie gilt im Mittel über Fälle und Kalibrierungen, nicht je Merkmalswert: die Schranke hat eine feste Breite für alle Fälle, eine Abhängigkeit der Breite von den Merkmalen ist nicht gemessen. |
| **Die Summe der Kantenschranken hat eine Garantie** | Nein, nicht für den gewählten Weg: Die Bogenquantile gelten für feste Bögen, der Weg ist vom Löser ausgewählt. Gemessen überdeckt die Summe in den neun Zellen der Seitenleiste {de(min(_all_a))} bis {de(max(_all_a))} % (Median über {n_seeds} Zufalls-Seeds, Ziel 90 %). |
| **Das Rauschen ist je Bogen unabhängig** | So ist es gemessen, und die Summe der Kantenschranken wird dadurch konservativ (die Fehler heben sich teilweise auf; sie überdeckt mehr als nötig und ist breiter); bei korreliertem Rauschen ist nichts gemessen. |
| **Ein Modell, eine Welt** | Nur ein lineares Modell (Kleinste Quadrate), nur das 5 × 5 Gitter, nur diese Kostenformel; ein auf den Regret trainiertes Modell (SPO+) ist nicht gemessen. |
| **Eine Entscheidungsregel** | Der Löser wählt den kürzesten Weg unter der Vorhersage; die Wahl unter mehreren Entscheidungsregeln, robuste Entscheidungen und Nebenbedingungen mit Wahrscheinlichkeit sind nicht Gegenstand. |
| **Eine obere Schranke** | Nur die obere Schranke für die Kosten, keine untere und keine Aussage über den Regret gegen den besten Weg. Die Wegkosten-Schranke überdeckt in den neun Zellen im Median {de(min(_all_c))} bis {de(max(_all_c))} %. |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben und gesucht.** Gegeben sind Trainingsfälle $(x_i, c_i)$ mit Merkmalen $x_i \in \mathbb{R}^p$ und beobachteten Bogenkosten $c_i \in \mathbb{R}^d$, ein daraus angepasstes Modell $\hat c(x)$ (Kleinste Quadrate), ein Löser, der zu Kosten $c$ einen kürzesten Weg $w(c)$ liefert, und $n$ Kalibrierfälle, die nicht im Training waren. Gesucht ist eine obere Schranke $U(x)$ mit $P\big(c^\top w(\hat c(x)) \le U(x)\big) \ge 1 - \alpha$.

| Symbol | Bedeutung |
|---|---|
| $p = 5$, $d = 40$ | Zahl der Merkmale, Zahl der Bögen im 5 × 5 Gitter |
| $c$, $\hat c$ | wahre und vorhergesagte Bogenkosten |
| $w = w(\hat c)$ | der Weg (0/1-Vektor über die Bögen), den der Löser bei $\hat c$ wählt; 8 Bögen, 70 mögliche Wege |
| $s$ | Score: tatsächliche minus vorhergesagte Kosten des gewählten Weges |
| $n$, $\alpha$, $k$ | Zahl der Kalibrierfälle, Fehlerniveau (0,1), Rang des Quantils |
| $s_{(k)}$ | der $k$-kleinste der $n$ Kalibrier-Scores |
| $m = 8$ | Zahl der Bögen eines Weges (Bonferroni-Teiler) |
| $B$, $\deg$, $e$ | Gewichtsmatrix der Welt, Grad der Wahrheit, Rauschstufe |

**Score und Schranke (Verfahren C).**

$$s = c^\top w(\hat c) - \hat c^\top w(\hat c), \qquad k = \lceil (n+1)(1-\alpha) \rceil, \qquad U = \hat c^\top w(\hat c) + s_{(k)}$$

Ist $k > n$, setzt man $U = \infty$: mit $n$ Werten ist höchstens das Niveau $n/(n+1)$ erreichbar (Handbeispiel: $n = 5$, Niveau 0,9, $k = 6$).

**Garantie.** Sind die $n$ Kalibrier-Scores und der Score $s_{n+1}$ eines neuen Falls austauschbar, so ist der Rang von $s_{n+1}$ unter allen $n + 1$ Werten bei Scores ohne Gleichstand gleichverteilt, und

$$P\big(s_{n+1} \le s_{(k)}\big) \ge \frac{k}{n+1} \ge 1 - \alpha$$

(mit Gleichstand gilt ≥ statt =). Die Wahrscheinlichkeit läuft über neue Fälle **und** über die Kalibrierung; für eine feste Kalibrierung ist die Überdeckung bei Scores ohne Gleichstand beta-verteilt mit den Parametern $k$ und $n + 1 - k$ (Experiment 4).

**Verfahren A und B.** Je Bogen $e$ ist $q_e$ der $k$-kleinste Wert von $c_e - \hat c_e$ über die Kalibrierfälle, bei A zum Niveau $1 - \alpha$, bei B zum Niveau $1 - \alpha / m$:

$$U_A = \hat c^\top w + \sum_{e:\, w_e = 1} q_e$$

Die Quantile $q_e$ gelten für einen festen Bogen; der Weg ist aber vom Löser aus $\hat c$ gewählt, deshalb folgt aus ihnen keine Garantie für $U_A$ oder $U_B$.

**Optimismus.** Mit den vorhergesagten Kosten $\hat z = \hat c^\top w(\hat c)$ und den tatsächlichen Kosten $z = c^\top w(\hat c)$ der Testfälle ist er $100 \cdot (\bar z - \bar{\hat z}) / \bar{\hat z}$ in Prozent; die Breite einer Schranke ist $100 \cdot (\bar U - \bar{\hat z}) / \bar{\hat z}$.

**Handbeispiel.** Scores $-0{,}5;\ 0{,}2;\ 0{,}9;\ 1{,}4;\ 2{,}0$, $n = 5$: Niveau $0{,}5$ gibt $k = \lceil 6 \cdot 0{,}5 \rceil = 3$ und die Schranke $0{,}9$; Niveau $0{,}8$ gibt $k = 5$ und $2{,}0$; Niveau $0{,}9$ gibt $k = 6 > 5$ und $\infty$.

**Welt.** $x \sim \mathcal N(0, I_p)$, $B \in \{0,1\}^{d \times p}$ mit unabhängigen fairen Münzwürfen, mittlere Kosten $m_j(x) = \big((Bx)_j / \sqrt{p} + 3\big)^{\deg} / 3{,}5^{\deg} + 1$, beobachtete Kosten $c_j = m_j(x)\,\varepsilon_j$ mit $\varepsilon_j \sim U[1 - e, 1 + e]$ unabhängig je Bogen und Fall.

**Literatur.** Konzeptquelle (ein einziges Papier, nur auf der Ebene des Abstracts gelesen, nicht nachgebaut): Cortes-Gomez et al., *Decision-Focused Uncertainty Quantification* (arXiv 2410.01767; in der zweiten Fassung umbenannt in *Utility-Directed Conformal Prediction*), bildet konforme Mengen unter Berücksichtigung des Entscheidungsverlusts bei erhaltener Abdeckungsgarantie. Die Idee, den Fehler nicht je Einzelwert, sondern als Summe über einen Zeitraum zu kalibrieren, steht schon in [forecast-inventory-demo](https://github.com/sebastian-hanisch/forecast-inventory-demo) (Sicherheitsbestand über das Schutzintervall); konforme Prognoseintervalle in [forecast-interval-demo](https://github.com/sebastian-hanisch/forecast-interval-demo). Neu hier: die Auswahl durch den Löser (Optimismus) und der Score auf dem Wert der gewählten Entscheidung. Das Gitter und die Kostenformel sind aus Stück 3 der Reihe (aus dem Gedächtnis nach dem Schema der SPO-Literatur nachgebaut, nicht gegen sie geprüft).
        """
    )

sync_query_params({
    "deg_slider": int(deg), "noise_slider": float(noise), "seed_input": int(seed), "case_slider": int(case), "level_slider": float(level),
})

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
