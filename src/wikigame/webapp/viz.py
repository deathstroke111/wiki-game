"""Plotly figures for the app. Colors are mid-tone so they read on both light and dark themes."""
from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go

COLORS = {"agent": "#EF553B", "optimal": "#00CC96", "neighbor": "#9AA0A6", "source": "#636EFA",
          "target": "#FFA15A", "wrong": "#AB63FA"}
MACHINE_COLORS = {"local": "#636EFA", "cuda": "#00CC96", "unknown": "#9AA0A6"}


def _node_style(kinds: set[str], wrong: bool) -> tuple[str, str, int]:
    if "source" in kinds:
        return COLORS["source"], "diamond", 20
    if "target" in kinds:
        return COLORS["target"], "star", 24
    if wrong:
        return COLORS["wrong"], "circle", 15
    if "agent" in kinds:
        return COLORS["agent"], "circle", 14
    if "optimal" in kinds:
        return COLORS["optimal"], "circle", 14
    return COLORS["neighbor"], "circle", 9


def path_figure(sub: dict, height: int = 520) -> go.Figure:
    """Graph of the agent path (red), optimal path (green) and sampled neighbours (grey)."""
    g = nx.Graph()
    g.add_nodes_from(sub["nodes"])
    g.add_edges_from((u, v) for u, v, _ in sub["edges"])
    pos = nx.spring_layout(g, seed=0, k=1.2 / max(1, len(g)) ** 0.5, iterations=200)

    fig = go.Figure()
    for kind, width, dash in (("neighbor", 1, "dot"), ("optimal", 3, "solid"), ("agent", 3, "solid")):
        xs, ys = [], []
        for u, v, k in sub["edges"]:
            if k == kind:
                xs += [pos[u][0], pos[v][0], None]
                ys += [pos[u][1], pos[v][1], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=True,
                                     name={"neighbor": "other links", "optimal": "optimal path",
                                           "agent": "agent path"}[kind],
                                     line={"color": COLORS[kind], "width": width, "dash": dash}))
    for u, v, k in sub["edges"]:  # arrowheads only on the two paths
        if k in ("agent", "optimal"):
            fig.add_annotation(x=pos[v][0], y=pos[v][1], ax=pos[u][0], ay=pos[u][1], xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.1,
                               arrowwidth=2, arrowcolor=COLORS[k], standoff=9, opacity=0.9)

    titles = list(sub["nodes"])
    styles = [_node_style(sub["nodes"][t]["kinds"], sub["nodes"][t]["wrong"]) for t in titles]
    fig.add_trace(go.Scatter(
        x=[pos[t][0] for t in titles], y=[pos[t][1] for t in titles], mode="markers+text",
        text=[t if sub["nodes"][t]["kinds"] - {"neighbor"} else "" for t in titles],
        textposition="top center", hovertext=titles, hoverinfo="text", showlegend=False,
        marker={"color": [s[0] for s in styles], "symbol": [s[1] for s in styles],
                "size": [s[2] for s in styles], "line": {"width": 1, "color": "rgba(255,255,255,0.7)"}}))
    # legend-only entries for node roles
    for label, color in (("source", COLORS["source"]), ("target", COLORS["target"]),
                         ("wrong turn", COLORS["wrong"])):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label, marker={"color": color, "size": 10}))
    fig.update_layout(height=height, margin={"l": 0, "r": 0, "t": 10, "b": 0}, xaxis={"visible": False},
                      yaxis={"visible": False}, legend={"orientation": "h", "y": -0.02},
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig
