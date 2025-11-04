# -*- coding: utf-8 -*-
"""
Adaptive Learning + Independent Learning Agent (with ChatGPT explanation)
Prototype consistent with "ARTICULO-CASO-01-INGLES..." and
"REDEFINING INDEPENDENT LEARNING WITH AI AGENTS"

Run:
    python app.py
Then open: http://127.0.0.1:8050

Tested for Dash >= 3.x and Spyder (use_reloader=False).

Versión con mejoras visuales (layout, colores, tarjetas, tabs) sin cambiar funcionalidad.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, dash_table
from dash.dependencies import ALL

# ✅ se asume instalada la librería openai
from openai import OpenAI

# ============================================================
# 0) THEME & STYLE CONFIG (mejorado)
# ============================================================

THEMES = {
    "light": {
        # Plotly base
        "template": "plotly_white",
        # Fondos
        "app_bg": "linear-gradient(160deg, #eef2ff 0%, #f9fafb 50%, #ffffff 100%)",
        "paper_bg": "rgba(255,255,255,1)",
        "plot_bg": "rgba(255,255,255,0)",
        # Tipografía
        "font_color": "#0f172a",
        "muted": "#6b7280",
        "accent": "#4f46e5",
        # Elementos
        "grad1": "linear-gradient(140deg, rgba(79,70,229,0.06) 0%, rgba(59,130,246,0.02) 90%)",
        "kpi_bg": "linear-gradient(140deg, rgba(255,255,255,1) 0%, rgba(248,250,252,1) 100%)",
        "card_bg": "#ffffff",
        "shadow": "0 14px 28px rgba(15, 23, 42, 0.04), 0 10px 10px rgba(15, 23, 42, 0.02)",
        "border": "1px solid rgba(148,163,184,0.15)",
    }
}

# Paleta para dominios
DOMAIN_COLORS = ["#4f46e5", "#22c55e", "#f97316", "#ef4444", "#8b5cf6"]
DOMAINS = ["Algebra", "Probability", "Calculus", "Statistics", "Programming"]


def card(children, theme="light", padding="16px", extra_style=None):
    th = THEMES[theme]
    base_style = {
        "background": th["card_bg"],
        "borderRadius": "20px",
        "padding": padding,
        "boxShadow": th["shadow"],
        "border": th["border"],
        "marginBottom": "18px",
        "overflow": "hidden",
        "backdropFilter": "blur(1.2px)",
    }
    if extra_style:
        base_style.update(extra_style)
    return html.Div(children, style=base_style)


def heading(text, theme="light", size="20px", marginBottom="8px", subtitle=None):
    th = THEMES[theme]
    return html.Div(
        [
            html.Div(
                text,
                style={
                    "fontWeight": "700",
                    "fontSize": size,
                    "color": th["font_color"],
                    "letterSpacing": "-0.01em",
                },
            ),
            html.Div(
                style={
                    "height": "3px",
                    "width": "52px",
                    "background": th["accent"],
                    "borderRadius": "999px",
                    "marginTop": "8px",
                }
            ),
            (html.Div(subtitle, style={"color": th["muted"], "marginTop": "6px", "fontSize": "13px"})
             if subtitle else None),
        ],
        style={"marginBottom": marginBottom},
    )


def kpi(title, value, theme="light", suffix=""):
    th = THEMES[theme]
    return html.Div(
        [
            html.Div(
                title.upper(),
                style={
                    "fontSize": "11px",
                    "color": th["muted"],
                    "marginBottom": "6px",
                    "letterSpacing": "0.06em",
                },
            ),
            html.Div(
                f"{value}{suffix}",
                style={"fontSize": "26px", "fontWeight": "700", "color": th["font_color"]},
            ),
        ],
        style={
            "background": th["kpi_bg"],
            "borderRadius": "16px",
            "padding": "14px 16px 14px 16px",
            "border": "1px solid rgba(148,163,184,0.14)",
            "minWidth": "150px",
        },
    )


def apply_figure_theme(fig, theme="light"):
    th = THEMES[theme]
    fig.update_layout(
        template=th["template"],
        paper_bgcolor=th["paper_bg"],
        plot_bgcolor=th["plot_bg"],
        font=dict(color=th["font_color"]),
    )
    return fig


# ============================================================
# 1) DATA SIMULATION
# ============================================================

np.random.seed(42)
N_STUDENTS = 90
N_ITEMS = 120
HORIZON_DAYS = 30


def simulate_students(n=N_STUDENTS, domains=DOMAINS):
    ability_g = np.random.normal(0, 1, n)
    engagement = np.clip(np.random.beta(3, 2, n), 0, 1)
    pace = np.clip(np.random.normal(45, 15, n), 5, 120)
    weekly_goal_hours = np.clip(np.random.normal(3.6, 1.2, n), 1.0, 8.0)

    df = pd.DataFrame(
        {
            "student_id": np.arange(n),
            "ability_g": ability_g,
            "engagement": engagement,
            "pace_min_day": pace,
            "goal_hours_week": weekly_goal_hours,
        }
    )

    for d in domains:
        df[f"affinity_{d}"] = np.clip(np.random.normal(0, 1, n), -2, 2)

    return df


def simulate_item_bank(n_items=N_ITEMS, domains=DOMAINS):
    domains_items = np.random.choice(domains, size=n_items, replace=True)
    difficulty_b = np.random.normal(0, 1.0, n_items)
    discrimination_a = np.clip(np.random.normal(1.0, 0.3, n_items), 0.6, 1.8)
    t_min = np.clip(np.random.normal(8, 3, n_items), 3, 20)

    df = pd.DataFrame(
        {
            "item_id": np.arange(n_items),
            "domain": domains_items,
            "a": discrimination_a,
            "b": difficulty_b,
            "t_expected_min": t_min,
        }
    )
    return df


def simulate_trajectories(df_students, domains=DOMAINS, days=HORIZON_DAYS):
    rows = []
    today = datetime.now().date()
    for _, r in df_students.iterrows():
        for d in domains:
            lam = max(5, r["pace_min_day"] * (0.5 + 0.5 * (r[f"affinity_{d}"] + 2) / 4))
            for k in range(days):
                dt = today - timedelta(days=days - k)
                minutes = np.random.poisson(lam=lam / 4.5)
                mu = (
                    0.55
                    + 0.18 * r["ability_g"]
                    + 0.12 * r[f"affinity_{d}"]
                    + np.random.normal(0, 0.08)
                )
                micro = float(np.clip(mu, 0.05, 0.98))
                rows.append([int(r["student_id"]), d, dt, int(minutes), micro])
    return pd.DataFrame(rows, columns=["student_id", "domain", "date", "minutes", "micro_score"])


DF_STUD = simulate_students()
DF_ITEMS = simulate_item_bank()
DF_TRAJ = simulate_trajectories(DF_STUD)


def recent_progress(df_traj, days=7):
    cutoff = (datetime.now().date() - timedelta(days=days))
    df = df_traj[df_traj["date"] >= cutoff]
    return df.groupby(["student_id", "domain"], as_index=False).agg(
        minutes=("minutes", "sum"),
        micro_mean=("micro_score", "mean")
    )


def p_success(a, b, theta):
    return 1 / (1 + np.exp(-a * (theta - b)))


def recommend_items(student_id, df_students, df_items, score_history):
    recs = []
    for d in DOMAINS:
        hist = score_history[
            (score_history.student_id == student_id)
            & (score_history.domain == d)
        ]
        if len(hist) > 0:
            theta_d = np.clip((hist.micro_score.mean() - 0.5) / 0.2, -2.5, 2.5)
        else:
            row = df_students.loc[df_students.student_id == student_id].iloc[0]
            theta_d = np.clip(
                row["ability_g"] * 0.7 + row[f"affinity_{d}"] * 0.3, -2.5, 2.5
            )
        candidates = df_items[df_items.domain == d].copy()
        candidates["gap"] = np.abs(candidates["b"] - theta_d)
        candidates["p_success"] = p_success(
            candidates["a"], candidates["b"], theta_d
        )
        top = candidates.sort_values("gap").head(3)
        top["theta_d"] = theta_d
        recs.append(top)
    return pd.concat(recs).sort_values(["domain", "gap"]).reset_index(drop=True)


# ============================================================
# 2) DASH APP LAYOUT (mejorado)
# ============================================================

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Adaptive Learning Prototype + ChatGPT"

student_options = [
    {"label": f"Student {i}", "value": int(i)} for i in DF_STUD["student_id"].tolist()
]
domain_options = [{"label": d, "value": d} for d in DOMAINS]

# Estilos globales de Tabs para que se vean más modernas
tabs_styles = {
    "width": "100%",
    "border": "none",
    "background": "transparent",
}
tab_style = {
    "padding": "8px 15px",
    "background": "transparent",
    "border": "none",
    "color": "#475569",
    "fontWeight": "500",
    "fontSize": "14px",
}
tab_selected_style = {
    "padding": "10px 15px",
    "background": "#ffffff",
    "border": "1px solid rgba(79,70,229,0.16)",
    "borderBottom": "3px solid #4f46e5",
    "borderRadius": "16px 16px 0 0",
    "color": "#0f172a",
    "boxShadow": "0 10px 16px rgba(15,23,42,0.05)",
}

app.layout = html.Div(
    [
        dcc.Store(id="store-api-key"),
        # CONTENEDOR PRINCIPAL
        html.Div(
            [
                # ==== TOPBAR ====
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    "Adaptive Learning Studio",
                                    style={
                                        "fontWeight": "700",
                                        "fontSize": "20px",
                                        "color": "#0f172a",
                                    },
                                ),
                                html.Div(
                                    "Prototype enhanced UI · Inclusive and AI-driven",
                                    style={
                                        "color": "#6b7280",
                                        "fontSize": "12px",
                                        "marginTop": "2px",
                                    },
                                ),
                            ],
                            style={"flex": "1"},
                        ),
                        html.Div(
                            [
                                dcc.Input(
                                    id="input-api-key",
                                    type="password",
                                    placeholder="sk-... API OpenAI ...",
                                    style={
                                        "width": "260px",
                                        "marginRight": "6px",
                                        "borderRadius": "12px",
                                        "border": "1px solid rgba(148,163,184,0.4)",
                                        "padding": "6px 12px",
                                        "outline": "none",
                                    },
                                ),
                                html.Button(
                                    "Guardar API key",
                                    id="btn-save-key",
                                    n_clicks=0,
                                    style={
                                        "background": "#4f46e5",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "7px 14px",
                                        "borderRadius": "12px",
                                        "cursor": "pointer",
                                        "fontWeight": "500",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "6px",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "padding": "12px 18px 10px 16px",
                        "background": "rgba(238,242,255,0.4)",
                        "backdropFilter": "blur(1.5px)",
                        "borderBottom": "1px solid rgba(148,163,184,0.12)",
                    },
                ),
                # ==== BODY ====
                html.Div(
                    [
                        # TABS
                        dcc.Tabs(
                            id="tabs",
                            value="tab_home",
                            children=[
                                dcc.Tab(
                                    label="Home",
                                    value="tab_home",
                                    style=tab_style,
                                    selected_style=tab_selected_style,
                                ),
                                dcc.Tab(
                                    label="Learning",
                                    value="tab_learning",
                                    style=tab_style,
                                    selected_style=tab_selected_style,
                                ),
                                dcc.Tab(
                                    label="Assessments",
                                    value="tab_assess",
                                    style=tab_style,
                                    selected_style=tab_selected_style,
                                ),
                                dcc.Tab(
                                    label="Analytics",
                                    value="tab_analytics",
                                    style=tab_style,
                                    selected_style=tab_selected_style,
                                ),
                                dcc.Tab(
                                    label="Agent",
                                    value="tab_agent",
                                    style=tab_style,
                                    selected_style=tab_selected_style,
                                ),
                                dcc.Tab(
                                    label="About",
                                    value="tab_about",
                                    style=tab_style,
                                    selected_style=tab_selected_style,
                                ),
                            ],
                            style=tabs_styles,
                        ),
                        # CONTENIDO
                        html.Div(
                            id="content",
                            style={
                                "padding": "18px 18px 22px 18px",
                                "minHeight": "calc(100vh - 110px)",
                            },
                        ),
                    ],
                    style={
                        "background": THEMES["light"]["app_bg"],
                        "minHeight": "100vh",
                    },
                ),
            ],
            style={"maxWidth": "1240px", "margin": "0 auto"},
        ),
    ]
)

# ============================================================
# 3) TAB RENDERING
# ============================================================

@app.callback(Output("content", "children"), Input("tabs", "value"))
def render_tab(tab):
    theme = "light"

    if tab == "tab_home":
        df7 = recent_progress(DF_TRAJ, days=7)
        total_minutes = int(df7["minutes"].sum())
        avg_perf = round(df7["micro_mean"].mean(), 3) if len(df7) else 0.0
        avg_goal = round(DF_STUD["goal_hours_week"].mean(), 1)

        return html.Div(
            [
                heading(
                    "Dashboard overview",
                    theme,
                    subtitle="Seguimiento sintético semanal de minutos, rendimiento y metas declaradas.",
                ),
                html.Div(
                    [
                        kpi("Total minutes (7d)", total_minutes, theme),
                        kpi("Avg. performance (7d)", avg_perf, theme),
                        kpi("Avg. weekly goal (h)", avg_goal, theme),
                    ],
                    style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                ),
                card(
                    [
                        html.H4(
                            "Students summary",
                            style={"marginBottom": "14px", "color": "#0f172a"},
                        ),
                        dash_table.DataTable(
                            columns=[
                                {"name": "student_id", "id": "student_id"},
                                {"name": "ability_g", "id": "ability_g"},
                                {"name": "engagement", "id": "engagement"},
                                {"name": "goal_hours_week", "id": "goal_hours_week"},
                            ],
                            data=DF_STUD.round(3).to_dict("records"),
                            page_size=10,
                            style_table={"overflowX": "auto"},
                            style_as_list_view=True,
                            style_cell={
                                "fontSize": "12px",
                                "padding": "6px",
                                "minWidth": "110px",
                            },
                            style_header={
                                "backgroundColor": "rgba(79,70,229,0.08)",
                                "fontWeight": "700",
                                "border": "none",
                            },
                        ),
                    ],
                    theme,
                    extra_style={"marginTop": "16px"},
                ),
            ]
        )

    elif tab == "tab_learning":
        return html.Div(
            [
                heading(
                    "Learning trajectory, recommendations and AI explanation",
                    theme,
                    subtitle="Visualización integrada de la curva de aprendizaje con sugerencias adaptativas y explicación generada por IA.",
                ),
                card(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Student",
                                            style={"fontWeight": "600", "fontSize": "13px"},
                                        ),
                                        dcc.Dropdown(
                                            id="lrn-stud",
                                            options=student_options,
                                            value=0,
                                            clearable=False,
                                        ),
                                        html.Br(),
                                        html.Label(
                                            "Domain",
                                            style={"fontWeight": "600", "fontSize": "13px"},
                                        ),
                                        dcc.Dropdown(
                                            id="lrn-dom",
                                            options=domain_options,
                                            value=DOMAINS[0],
                                            clearable=False,
                                        ),
                                        html.Br(),
                                        html.Button(
                                            "Compute recommendations",
                                            id="btn-recommend",
                                            n_clicks=0,
                                            style={
                                                "background": "#4f46e5",
                                                "color": "white",
                                                "border": "none",
                                                "padding": "7px 12px",
                                                "borderRadius": "10px",
                                                "cursor": "pointer",
                                                "fontWeight": "500",
                                            },
                                        ),
                                    ],
                                    style={
                                        "flex": "0 0 240px",
                                        "background": "rgba(79,70,229,0.03)",
                                        "borderRadius": "14px",
                                        "padding": "10px 10px 10px 10px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            dcc.Graph(
                                                id="fig-trajectory",
                                                style={"height": "320px"},
                                            ),
                                            style={
                                                "height": "320px",
                                                "overflow": "hidden",
                                                "marginBottom": "12px",
                                            },
                                        ),
                                        html.Div(id="recs-table"),
                                        html.Div(
                                            id="recs-explanation",
                                            style={
                                                "marginTop": "10px",
                                                "color": "#374151",
                                                "background": "rgba(248,250,252,0.4)",
                                                "padding": "8px 10px",
                                                "borderRadius": "10px",
                                            },
                                        ),
                                    ],
                                    style={"flex": "1", "paddingLeft": "14px"},
                                ),
                            ],
                            style={"display": "flex", "flexWrap": "wrap", "gap": "14px"},
                        )
                    ],
                    theme,
                ),
            ]
        )

    elif tab == "tab_assess":
        return html.Div(
            [
                heading(
                    "Adaptive assessments (simple IRT-like)",
                    theme,
                    subtitle="Se generan ítems ajustados al nivel de desempeño esperado y se recopilan las respuestas.",
                ),
                card(
                    [
                        html.Div(
                            [
                                html.Label("Student", style={"fontWeight": "600"}),
                                dcc.Dropdown(
                                    id="assess-stud",
                                    options=student_options,
                                    value=0,
                                    clearable=False,
                                ),
                                html.Br(),
                                html.Label("Domain", style={"fontWeight": "600"}),
                                dcc.Dropdown(
                                    id="assess-dom",
                                    options=domain_options,
                                    value=DOMAINS[0],
                                    clearable=False,
                                ),
                                html.Br(),
                                html.Label(
                                    "Number of questions", style={"fontWeight": "600"}
                                ),
                                dcc.Slider(
                                    id="assess-nq",
                                    min=3,
                                    max=10,
                                    step=1,
                                    value=5,
                                    marks={i: str(i) for i in range(3, 11)},
                                ),
                                html.Br(),
                                html.Button(
                                    "Generate assessment",
                                    id="btn-generate-assess",
                                    n_clicks=0,
                                    style={
                                        "background": "#4f46e5",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "7px 12px",
                                        "borderRadius": "10px",
                                        "cursor": "pointer",
                                        "fontWeight": "500",
                                    },
                                ),
                            ],
                            style={
                                "flex": "0 0 300px",
                                "background": "rgba(79,70,229,0.03)",
                                "borderRadius": "14px",
                                "padding": "10px",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(id="assess-questions"),
                                html.Br(),
                                html.Button(
                                    "Submit answers",
                                    id="btn-submit-assess",
                                    n_clicks=0,
                                    style={
                                        "background": "#22c55e",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "7px 12px",
                                        "borderRadius": "10px",
                                        "cursor": "pointer",
                                        "fontWeight": "500",
                                    },
                                ),
                                html.Div(
                                    id="assess-feedback", style={"marginTop": "12px"}
                                ),
                            ],
                            style={"flex": "1", "paddingLeft": "16px"},
                        ),
                    ],
                    theme,
                ),
            ]
        )

    elif tab == "tab_analytics":
        df7 = recent_progress(DF_TRAJ, days=7)
        fig_hm = px.density_heatmap(
            df7,
            x="domain",
            y="student_id",
            z="micro_mean",
            histfunc="avg",
            color_continuous_scale="Viridis",
            title="7-day performance heatmap",
        )
        apply_figure_theme(fig_hm, theme)
        fig_hm.update_layout(height=360, margin=dict(l=40, r=10, t=50, b=30))

        fig_sc = px.scatter(
            DF_STUD,
            x="engagement",
            y="ability_g",
            color=DF_STUD.index,
            title="Ability vs Engagement",
            labels={"engagement": "Engagement", "ability_g": "Ability (g)"},
        )
        apply_figure_theme(fig_sc, theme)
        fig_sc.update_layout(height=320, margin=dict(l=30, r=10, t=40, b=30))

        fig_hist = px.histogram(
            DF_STUD, x="ability_g", nbins=25, title="Distribution of ability (g)"
        )
        apply_figure_theme(fig_hist, theme)
        fig_hist.update_layout(height=320, margin=dict(l=30, r=10, t=40, b=30))

        return html.Div(
            [
                heading(
                    "Learning analytics",
                    theme,
                    subtitle="Relación entre compromiso, capacidad y resultados recientes.",
                ),
                card([dcc.Graph(figure=fig_hm, style={"height": "360px"})], theme),
                html.Div(
                    [
                        card(
                            [dcc.Graph(figure=fig_hist, style={"height": "320px"})],
                            theme,
                            padding="10px",
                        ),
                        card(
                            [dcc.Graph(figure=fig_sc, style={"height": "320px"})],
                            theme,
                            padding="10px",
                        ),
                    ],
                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                ),
            ]
        )

    elif tab == "tab_agent":
        return html.Div(
            [
                heading(
                    "Independent Learning Agent (weekly plan)",
                    theme,
                    subtitle="Distribución semanal de horas según prioridad formativa y evidencia reciente.",
                ),
                card(
                    [
                        html.Div(
                            [
                                html.Label("Student", style={"fontWeight": "600"}),
                                dcc.Dropdown(
                                    id="ag-stud",
                                    options=student_options,
                                    value=0,
                                    clearable=False,
                                ),
                                html.Br(),
                                html.Label("Priority", style={"fontWeight": "600"}),
                                dcc.Dropdown(
                                    id="ag-priority",
                                    options=[
                                        {
                                            "label": "Reinforce weaknesses",
                                            "value": "weak",
                                        },
                                        {
                                            "label": "Consolidate strengths",
                                            "value": "strong",
                                        },
                                        {"label": "Balanced", "value": "balanced"},
                                    ],
                                    value="weak",
                                    clearable=False,
                                ),
                                html.Br(),
                                html.Label(
                                    "Weekly goal (hours)", style={"fontWeight": "600"}
                                ),
                                dcc.Slider(
                                    id="ag-goal",
                                    min=2,
                                    max=10,
                                    step=0.5,
                                    value=4.0,
                                    marks={i: str(i) for i in range(2, 11)},
                                ),
                                html.Br(),
                                html.Button(
                                    "Generate agent plan",
                                    id="btn-agent",
                                    n_clicks=0,
                                    style={
                                        "background": "#4f46e5",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "7px 12px",
                                        "borderRadius": "10px",
                                        "cursor": "pointer",
                                        "fontWeight": "500",
                                    },
                                ),
                            ],
                            style={
                                "flex": "0 0 260px",
                                "background": "rgba(79,70,229,0.03)",
                                "borderRadius": "14px",
                                "padding": "10px",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    id="agent-text",
                                    style={
                                        "marginBottom": "14px",
                                        "maxHeight": "120px",
                                        "overflow": "auto",
                                    },
                                ),
                                dcc.Graph(id="agent-fig", style={"height": "340px"}),
                            ],
                            style={"flex": "1", "paddingLeft": "16px"},
                        ),
                    ],
                    theme,
                ),
            ]
        )

    else:  # tab_about
        return html.Div(
            [
                heading("About this prototype", theme),
                card(
                    [
                        html.P(
                            "This prototype demonstrates an AI-based adaptive learning web application built with Python, "
                            "Plotly, and Dash. It simulates students, learning trajectories, item banks, and AI-driven "
                            "recommendations. The system includes: (1) dashboards, (2) learning trajectories and recommendations, "
                            "(3) a simple adaptive assessment engine, (4) cohort analytics, and (5) an independent learning agent "
                            "that allocates weekly study time. A ChatGPT endpoint can be used to explain why each recommendation "
                            "is proposed, thus turning analytics into pedagogical guidance.",
                            style={"fontSize": "14px", "lineHeight": "1.6"},
                        )
                    ],
                    theme,
                ),
            ]
        )


# ============================================================
# 4) SAVE API KEY
# ============================================================

@app.callback(
    Output("store-api-key", "data"),
    Input("btn-save-key", "n_clicks"),
    State("input-api-key", "value"),
    prevent_initial_call=True,
)
def save_api_key(n, key):
    if not key:
        return {}
    return {"api_key": key}


# ============================================================
# 5) LEARNING CALLBACK WITH ChatGPT (active)
# ============================================================

@app.callback(
    Output("fig-trajectory", "figure"),
    Output("recs-table", "children"),
    Output("recs-explanation", "children"),
    Input("btn-recommend", "n_clicks"),
    State("lrn-stud", "value"),
    State("lrn-dom", "value"),
    State("store-api-key", "data"),
    prevent_initial_call=True,
)
def update_learning(n, student_id, dom, api_store):
    theme = "light"

    tr = DF_TRAJ[
        (DF_TRAJ.student_id == student_id) & (DF_TRAJ.domain == dom)
    ].sort_values("date")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tr["date"],
            y=tr["minutes"],
            name="Minutes",
            marker_color="#4f46e5",
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tr["date"],
            y=tr["micro_score"],
            name="Micro-score",
            yaxis="y2",
            line=dict(width=3, color="#22c55e"),
        )
    )
    fig.update_layout(
        title=f"Trajectory – Student {student_id} | {dom}",
        yaxis=dict(title="Minutes"),
        yaxis2=dict(
            title="Micro-score", overlaying="y", side="right", range=[0, 1.05]
        ),
        height=320,
        margin=dict(l=40, r=10, t=40, b=30),
        uirevision="const",
    )
    apply_figure_theme(fig, theme)

    recs = recommend_items(student_id, DF_STUD, DF_ITEMS, DF_TRAJ)
    recs_show = recs[
        ["domain", "item_id", "a", "b", "theta_d", "p_success", "t_expected_min"]
    ].copy().round(3)
    table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in recs_show.columns],
        data=recs_show.to_dict("records"),
        page_size=12,
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "12px", "padding": "6px"},
        style_header={"backgroundColor": "#eef2ff", "fontWeight": "700"},
    )

    if not api_store or "api_key" not in api_store or not api_store["api_key"]:
        return (
            fig,
            table,
            html.Div(
                "Enter your OpenAI API key at the top to get an AI-generated explanation for these recommendations.",
                style={"fontStyle": "italic", "color": "#6b7280"},
            ),
        )

    api_key = api_store["api_key"]

    prompt = f"""
You are an educational AI tutor. A learner with id {student_id} is studying the domain {dom}.
You have the following recommended items, each with IRT-like parameters:

{recs_show.to_markdown(index=False)}

Explain in one paragraph, clear, in academic English, why these items were recommended, referring to:
- the match between difficulty b and the estimated theta_d,
- keeping the student in an optimal challenge zone,
- and prioritizing weaker domains.
"""

    explanation_text = (
        "These items were selected because their difficulty parameters are close to the learner’s current estimated "
        "competence (theta_d), which maintains the student in a productive challenge zone."
    )

    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a pedagogical explainer for adaptive learning systems.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=180,
        )
        explanation_text = resp.choices[0].message.content.strip()
    except Exception as e:
        explanation_text = f"AI explanation could not be retrieved: {str(e)}"

    explanation_div = html.Div(
        [html.Strong("AI explanation: "), html.Span(explanation_text)]
    )

    return fig, table, explanation_div


# ============================================================
# 6) ASSESSMENTS CALLBACKS
# ============================================================

@app.callback(
    Output("assess-questions", "children"),
    Input("btn-generate-assess", "n_clicks"),
    State("assess-stud", "value"),
    State("assess-dom", "value"),
    State("assess-nq", "value"),
    prevent_initial_call=True,
)
def generate_assessment(n, student_id, dom, n_q):
    recs = recommend_items(student_id, DF_STUD, DF_ITEMS, DF_TRAJ)
    dom_items = recs[recs["domain"] == dom].head(int(n_q * 2))
    dom_items = dom_items.head(n_q)

    questions_children = []
    for i, (_, row) in enumerate(dom_items.iterrows()):
        question_text = (
            f"Q{i+1}: Practice item {int(row['item_id'])} ({dom}) – expected difficulty b={row['b']:.2f}"
        )
        questions_children.append(
            html.Div(
                [
                    html.Div(question_text, style={"marginBottom": "4px"}),
                    dcc.RadioItems(
                        id={"type": "assess-answer", "index": i},
                        options=[
                            {"label": "Correct", "value": 1},
                            {"label": "Incorrect", "value": 0},
                        ],
                        value=None,
                        inline=True,
                    ),
                ],
                style={
                    "marginBottom": "10px",
                    "background": "rgba(248,250,252,1)",
                    "padding": "6px 10px",
                    "borderRadius": "10px",
                },
            )
        )

    return html.Div(questions_children)


@app.callback(
    Output("assess-feedback", "children"),
    Input("btn-submit-assess", "n_clicks"),
    State({"type": "assess-answer", "index": ALL}, "value"),
    State("assess-stud", "value"),
    prevent_initial_call=True,
)
def submit_assessment(n, answers, student_id):
    if not answers:
        return html.Div("No answers submitted.", style={"color": "#ef4444"})
    answers_clean = [a for a in answers if a is not None]
    if len(answers_clean) == 0:
        return html.Div("Please answer at least one question.", style={"color": "#ef4444"})
    score = sum(answers_clean) / len(answers_clean)
    return html.Div(
        f"Assessment submitted. Accuracy: {score*100:.1f}%.",
        style={"color": "#22c55e", "fontWeight": "600"},
    )


# ============================================================
# 7) AGENT CALLBACK
# ============================================================

@app.callback(
    Output("agent-text", "children"),
    Output("agent-fig", "figure"),
    Input("btn-agent", "n_clicks"),
    State("ag-stud", "value"),
    State("ag-priority", "value"),
    State("ag-goal", "value"),
    prevent_initial_call=True,
)
def build_agent_plan(n, student_id, priority, goal_h):
    theme = "light"
    df7 = recent_progress(DF_TRAJ, days=7)
    df7s = df7[df7["student_id"] == student_id].copy()
    if len(df7s) == 0:
        weights = pd.DataFrame(
            {"domain": DOMAINS, "weight": np.ones(len(DOMAINS)) / len(DOMAINS)}
        )
    else:
        if priority == "weak":
            df7s["inv"] = 1 - df7s["micro_mean"]
            weights = df7s[["domain", "inv"]].rename(columns={"inv": "weight"})
        elif priority == "strong":
            df7s["wt"] = df7s["micro_mean"]
            weights = df7s[["domain", "wt"]].rename(columns={"wt": "weight"})
        else:
            weights = df7s[["domain"]].copy()
            weights["weight"] = 1.0
    weights = weights.groupby("domain", as_index=False).agg(weight=("weight", "mean"))
    weights["weight"] = weights["weight"] / weights["weight"].sum()

    engagement = float(
        DF_STUD.loc[DF_STUD.student_id == student_id, "engagement"].iloc[0]
    )
    weights["hours"] = weights["weight"] * goal_h

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rows = []
    for d in weights["domain"]:
        base = weights.loc[weights.domain == d, "hours"].iloc[0]
        day_weights = (
            np.array([1.1, 1.1, 1.1, 1.0, 1.0, 0.8, 0.9]) * (0.8 + 0.4 * engagement)
        )
        day_weights = day_weights / day_weights.sum()
        assign = base * day_weights
        for i, day in enumerate(days):
            rows.append([d, day, float(assign[i])])

    df_sched = pd.DataFrame(rows, columns=["domain", "day", "hours"])
    fig = px.bar(
        df_sched,
        x="day",
        y="hours",
        color="domain",
        barmode="stack",
        title=f"Agent weekly plan (goal {goal_h:.1f} h)",
        color_discrete_sequence=DOMAIN_COLORS,
    )
    apply_figure_theme(fig, theme)
    fig.update_traces(marker_line_width=0)
    fig.update_layout(height=340, margin=dict(l=40, r=10, t=40, b=30))

    explanation = html.Div(
        [
            html.B("Plan rationale: "),
            html.Span("priority "),
            html.Code(priority),
            html.Span(
                " adjusts weights using recent performance; hours are distributed by factoring engagement to favor consistency. "
                "If recent data are scarce, a balanced plan is proposed."
            ),
            html.Br(),
            html.Br(),
            html.B("Micro-recommendations: "),
            html.Ul(
                [
                    html.Li(
                        "Begin sessions with 10–15 minutes of guided review of the lowest-scoring domain."
                    ),
                    html.Li(
                        "Interleave items with b≈θ and b≈θ+0.3 to sustain optimal challenge."
                    ),
                    html.Li("End with a short reflection about what was learned."),
                ]
            ),
        ]
    )

    return explanation, fig


# ============================================================
# 8) MAIN
# ============================================================

if __name__ == "__main__":
    # Se deja igual que el original
    app.run(host="127.0.0.1", port=8050, debug=False, use_reloader=False)


