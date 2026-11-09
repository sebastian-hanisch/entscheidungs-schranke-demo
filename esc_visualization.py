"""Plotly-Abbildungen (alle Achsen gesperrt, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

import esc_constants as C
from esc_evaluation import de

_LOCK = dict(fixedrange=True)
_NAMES = {"a": "Summe der Kantenschranken", "b": "Summe mit Bonferroni", "c": "Konform auf den Wegkosten"}
_COLORS = {"a": C.COLOR_SUM, "b": C.COLOR_BONF, "c": C.COLOR_PATH}


def _lock(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(**_LOCK)
    fig.update_yaxes(**_LOCK)
    return fig


def _err(rows, key):
    """Fehlerbalken von Minimum bis Maximum um den Median."""
    return dict(type="data", symmetric=False, array=[r[key]["max"] - r[key]["median"] for r in rows], arrayminus=[r[key]["median"] - r[key]["min"] for r in rows])


# --- Experiment 1: der Score von Hand --------------------------------------------------------------------------------------------------------

def build_hand_figure(h: dict) -> go.Figure:
    """Die Kalibrierwerte auf einer Zahlengeraden mit ihrem Rang; die Schranke (k-kleinster Wert) als senkrechte Linie, bei k > n ein Hinweis statt der Linie."""
    scores = list(h["sorted"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scores, y=[0] * len(scores), mode="markers+text", text=[f"Rang {i + 1}" for i in range(len(scores))], textposition="top center",
                             marker=dict(size=13, color=C.COLOR_PRED), showlegend=False, hovertemplate="Score %{x:.1f}<extra></extra>"))
    lo, hi = min(scores) - 1.0, max(scores) + 1.5
    if h["infinite"]:
        fig.add_annotation(x=hi, y=0, text="Schranke unendlich (Rang k > n)", showarrow=False, xanchor="right", yshift=-34, font=dict(color=C.COLOR_REAL))
    else:
        fig.add_vline(x=h["q"], line=dict(color=C.COLOR_PATH, width=3, dash="dash"))
        fig.add_annotation(x=h["q"], y=0, text=f"Schranke {de(h['q'])}", showarrow=False, yshift=-34, font=dict(color=C.COLOR_PATH))
    fig.update_xaxes(title_text="Score = tatsächliche minus vorhergesagte Kosten des gewählten Weges", range=[lo, hi], zeroline=True)
    fig.update_yaxes(visible=False, range=[-1, 1])
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=50))
    return _lock(fig)


# --- Experiment 2: drei Schranken ---------------------------------------------------------------------------------------------------------------

def build_bounds_figure(rows: list[dict], live: dict | None = None) -> go.Figure:
    """Links: Überdeckung je Verfahren (Median, Balken von Minimum bis Maximum über die Zufalls-Seeds, Ziel als Linie); rechts: Breite in Prozent der Vorhersagekosten.
    `live`: Werte dieser einen Instanz (Rauten), Schlüssel cov_a ... w_c."""
    names = [r["Verfahren"] for r in rows]
    colors = [_COLORS[r["key"]] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Überdeckung (%)", "Breite (% der Vorhersage)"), horizontal_spacing=0.12)
    fig.add_trace(go.Bar(x=names, y=[r["coverage"]["median"] for r in rows], marker_color=colors, error_y=_err(rows, "coverage"), showlegend=False,
                         hovertemplate="%{x}: %{y:.1f} %<extra></extra>"), row=1, col=1)
    fig.add_hline(y=C.TARGET_PERCENT, line=dict(color="black", dash="dash"), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=[r["width"]["median"] for r in rows], marker_color=colors, error_y=_err(rows, "width"), showlegend=False,
                         hovertemplate="%{x}: %{y:.1f} %<extra></extra>"), row=1, col=2)
    if live:
        for col, prefix in ((1, "cov_"), (2, "w_")):
            fig.add_trace(go.Scatter(x=names, y=[live[prefix + r["key"]] for r in rows], mode="markers", marker=dict(color="black", size=10, symbol="diamond"),
                                     name="diese Welt (ein Zufalls-Seed)", showlegend=(col == 1), legendgroup="live", hovertemplate="%{x}: %{y:.1f} %<extra>diese Welt</extra>"), row=1, col=col)
    fig.update_yaxes(range=[0, 105], row=1, col=1)
    fig.update_layout(height=360, margin=dict(l=40, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.2, x=0.0))
    return _lock(fig)


def build_grid_figure(grid, chat: np.ndarray, w_chosen: np.ndarray, vmin: float, vmax: float) -> go.Figure:
    """5 x 5 Gitter: Bögen nach den VORHERGESAGTEN Kosten gefärbt und beschriftet (Farbe und Zahl: violett billig, gelb teuer), der vom Löser gewählte Weg dick."""
    h = grid.h
    span = max(vmax - vmin, 1e-9)
    colors = sample_colorscale(C.EDGE_COLORSCALE, [float(np.clip((c - vmin) / span, 0, 1)) for c in chat])
    fig = go.Figure()
    mids_x, mids_y, hover = [], [], []
    for k, (a, b) in enumerate(grid.edges):
        (x0, y0), (x1, y1) = grid.node_xy(a), grid.node_xy(b)
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=colors[k], width=3), hoverinfo="skip", showlegend=False))
        mids_x.append((x0 + x1) / 2)
        mids_y.append((y0 + y1) / 2)
        hover.append(f"Bogen {k + 1}: vorhergesagte Kosten {de(chat[k], 2)}")
    xs, ys = [], []
    for k in np.flatnonzero(w_chosen):
        a, b = grid.edges[k]
        (x0, y0), (x1, y1) = grid.node_xy(a), grid.node_xy(b)
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_PATH, width=8), name="gewählter Weg", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=mids_x, y=mids_y, mode="markers", marker=dict(size=14, opacity=0), hovertext=hover, hoverinfo="text", showlegend=False))
    nx = [grid.node_xy(v)[0] for v in range(grid.nodes)]
    ny = [grid.node_xy(v)[1] for v in range(grid.nodes)]
    fig.add_trace(go.Scatter(x=nx, y=ny, mode="markers", marker=dict(size=9, color="#333", line=dict(color="white", width=1)), hoverinfo="skip", showlegend=False))
    for k in range(len(mids_x)):
        fig.add_annotation(x=mids_x[k], y=mids_y[k], text=de(chat[k], 1), showarrow=False, font=dict(size=9), bgcolor="rgba(255,255,255,0.88)", bordercolor="rgba(0,0,0,0)")
    fig.add_annotation(x=0, y=0, text="Start", showarrow=False, xshift=-22, yshift=-12, font=dict(size=11))
    fig.add_annotation(x=h - 1, y=h - 1, text="Ziel", showarrow=False, xshift=20, yshift=12, font=dict(size=11))
    fig.update_xaxes(visible=False, range=[-0.35, h - 0.65])
    fig.update_yaxes(visible=False, range=[h - 0.45, -0.45])
    fig.update_layout(height=380, margin=dict(l=30, r=30, t=5, b=5), legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center", yanchor="top"))
    return _lock(fig)


def build_case_figure(c: dict) -> go.Figure:
    """Ein Testfall: vorhergesagte und tatsächliche Kosten des gewählten Weges und die drei Schranken als Balken."""
    labels = ["vorhergesagt", "tatsächlich", "Schranke Wegkosten", "Schranke Summe", "Schranke Bonferroni"]
    values = [c["pred"], c["real"], c["c"], c["a"], c["b"]]
    colors = [C.COLOR_PRED, C.COLOR_REAL, C.COLOR_PATH, C.COLOR_SUM, C.COLOR_BONF]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=[de(v, 1) for v in values], textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    fig.update_yaxes(title_text="Kosten des gewählten Weges", rangemode="tozero")
    fig.update_layout(height=380, margin=dict(l=40, r=20, t=30, b=40))
    return _lock(fig)


# --- Experiment 3: Optimismus der Auswahl ------------------------------------------------------------------------------------------------------

def build_optimism_figure(rows: list[dict]) -> go.Figure:
    """Je Grad: Optimismus (tatsächliche minus vorhergesagte Kosten, % der vorhergesagten) des gewählten Weges und, als Kontrolle, eines zufällig gewählten Weges (Median, Spanne über Seeds)."""
    x = [f"Grad {r['deg']}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[r["chosen"]["median"] for r in rows], name="gewählter Weg (Löser)", marker_color=C.COLOR_REAL, error_y=_err(rows, "chosen"),
                         hovertemplate="%{x}: %{y:.2f} %<extra>gewählter Weg</extra>"))
    fig.add_trace(go.Bar(x=x, y=[r["random"]["median"] for r in rows], name="zufällig gewählter Weg (Kontrolle)", marker_color=C.COLOR_PRED, error_y=_err(rows, "random"),
                         hovertemplate="%{x}: %{y:.2f} %<extra>zufälliger Weg</extra>"))
    fig.update_yaxes(title_text="Optimismus (% der Vorhersage)", zeroline=True, zerolinecolor="black")
    fig.update_layout(height=360, barmode="group", margin=dict(l=40, r=20, t=30, b=40), legend=dict(orientation="h", y=-0.2, x=0.0))
    return _lock(fig)


# --- Experiment 4: Kalibrierfälle ---------------------------------------------------------------------------------------------------------------

def build_blocks_figure(rows: list[dict]) -> go.Figure:
    """Links: Überdeckung der Wegkosten-Schranke je Zahl der Kalibrierfälle (Mittel, Balken vom 5. bis 95. Perzentil über die Kalibrierungen, Ziel als Linie); rechts: Breite (Median, Minimum bis Maximum)."""
    x = [f"n = {r['n']}" for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Überdeckung (%)", "Breite (% der Vorhersage)"), horizontal_spacing=0.12)
    cov = [r["coverage"] for r in rows]
    fig.add_trace(go.Scatter(x=x, y=[c["mean"] for c in cov], mode="markers", marker=dict(color=C.COLOR_PATH, size=11),
                             error_y=dict(type="data", symmetric=False, array=[c["p95"] - c["mean"] for c in cov], arrayminus=[c["mean"] - c["p05"] for c in cov], color=C.COLOR_PATH),
                             showlegend=False, hovertemplate="%{x}: Mittel %{y:.2f} %<extra></extra>"), row=1, col=1)
    fig.add_hline(y=C.TARGET_PERCENT, line=dict(color="black", dash="dash"), row=1, col=1)
    wid = [r["width"] for r in rows]
    fig.add_trace(go.Scatter(x=x, y=[w["median"] for w in wid], mode="markers", marker=dict(color=C.COLOR_PATH, size=11),
                             error_y=dict(type="data", symmetric=False, array=[w["max"] - w["median"] for w in wid], arrayminus=[w["median"] - w["min"] for w in wid], color=C.COLOR_PATH),
                             showlegend=False, hovertemplate="%{x}: Median %{y:.1f} %<extra></extra>"), row=1, col=2)
    fig.update_yaxes(range=[50, 102], row=1, col=1)
    fig.update_yaxes(rangemode="tozero", row=1, col=2)
    fig.update_layout(height=340, margin=dict(l=40, r=20, t=50, b=40))
    return _lock(fig)

