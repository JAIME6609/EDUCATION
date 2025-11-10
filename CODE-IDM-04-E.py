# =========================================================
# Intelligent Digital Mentors — Adaptive Learning Dashboard
# Full code in English with detailed inline comments
# Technologies: Python, Dash, Plotly, (optional) OpenAI API
# =========================================================

import os
from datetime import datetime
import random

import dash
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px

# =========================================================
# 1) TRY TO LOAD OPENAI CLIENT SAFELY
# ---------------------------------------------------------
# The application can work even without OpenAI (chat falls
# back to controlled messages). This block checks whether
# the openai client is available in the environment.
# =========================================================
try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OpenAI = None
    OPENAI_OK = False

# =========================================================
# 2) BASE DATA AND CONSTANTS
# ---------------------------------------------------------
# - DEFAULT_OPENAI_API_KEY reads from environment variables.
# - INITIAL_STUDENTS defines a simple simulated cohort with
#   progress and autonomy levels.
# - RESOURCES_BY_LEVEL is a rule-based recommender that
#   returns suggested resources according to autonomy.
# - MOD_COLOR_MAP provides persistent colors per module.
# =========================================================
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Initial cohort (simulated). Values in [0,1] for progress/autonomy.
INITIAL_STUDENTS = [
    {
        "id": 1,
        "name": "Student A",
        "module": "Linear Algebra",
        "progress": 0.35,
        "autonomy": 0.40,
        "difficulty": "medium",
    },
    {
        "id": 2,
        "name": "Student B",
        "module": "Programming I",
        "progress": 0.70,
        "autonomy": 0.65,
        "difficulty": "low",
    },
    {
        "id": 3,
        "name": "Student C",
        "module": "Differential Calculus",
        "progress": 0.50,
        "autonomy": 0.45,
        "difficulty": "high",
    },
    {
        "id": 4,
        "name": "Student D",
        "module": "Educational Statistics",
        "progress": 0.85,
        "autonomy": 0.80,
        "difficulty": "low",
    },
]

# Rule-based resource suggestions per autonomy level (0–100%)
RESOURCES_BY_LEVEL = {
    "very_low": [
        "5-minute introductory video about the topic.",
        "Infographic with basic steps.",
        "Guided activity with solved examples.",
    ],
    "low": [
        "Short reading with examples.",
        "Immediate self-assessment quiz.",
        "Visual summary with key terms.",
    ],
    "medium": [
        "Case applied to a real-world context.",
        "Problems of medium complexity.",
        "Collaborative activity with feedback.",
    ],
    "high": [
        "Open project where the student defines the problem.",
        "Advanced academic reading.",
        "Self-assessment and metacognition rubric.",
    ],
}

# Persistent color palette by module (improves visual consistency)
MOD_COLOR_MAP = {
    "Linear Algebra": "#38bdf8",
    "Programming I": "#8b5cf6",
    "Differential Calculus": "#f97316",
    "Educational Statistics": "#10b981",
}

# =========================================================
# 3) SAFE OPENAI CALL
# ---------------------------------------------------------
# This helper centralizes the interaction with OpenAI:
# - It gracefully handles missing library or missing API key.
# - It returns controlled strings if something goes wrong,
#   so the Dash callback won't crash.
# - It expects a "history" list in Chat Completions format.
# =========================================================
def safe_openai_call(api_key: str, system_instruction: str, history: list):
    """
    Call OpenAI's Chat Completions. If anything fails, return
    a safe, controlled message instead of raising exceptions.
    """
    if not OPENAI_OK:
        return (
            "The environment does not have the OpenAI library installed. "
            "Please install it using 'pip install openai'."
        )
    if not api_key:
        return (
            "No API Key received. Please enter your API Key at the top and resend your question."
        )
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            temperature=0.55,
            max_tokens=350,
        )
        if not resp or not resp.choices or not resp.choices[0].message:
            return "The API responded without content. Try again."
        return resp.choices[0].message.content
    except Exception as e:
        return (
            "The digital mentor is active with OpenAI, but the call failed.\n"
            f"Details: {str(e)}\n"
            "You can try again or check your key."
        )

# =========================================================
# 4) DASH APPLICATION: LAYOUT
# ---------------------------------------------------------
# The interface is organized in:
# - A sticky navbar with API key input.
# - Left panel: chat with the digital mentor.
# - Right panel: analytics (cards + filters + graphs).
# - Footer: informational note.
# Several dcc.Store components keep application state.
# =========================================================
app = Dash(__name__, suppress_callback_exceptions=True, title="Intelligent Digital Mentors - Enhanced")

app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "background": "linear-gradient(135deg, #0f172a 0%, #1e293b 35%, #f8fafc 100%)",
        "fontFamily": "'Segoe UI', sans-serif",
    },
    children=[
        # ---------------------------
        # Local stores (client-side)
        # ---------------------------
        dcc.Store(id="store-api-key", data={"api_key": DEFAULT_OPENAI_API_KEY}),
        dcc.Store(id="store-conversation", data=[]),
        dcc.Store(id="store-students", data=INITIAL_STUDENTS),
        dcc.Store(id="store-autonomy-log", data=[]),

        # Interval to simulate time-based updates (progress/autonomy)
        dcc.Interval(id="interval-update", interval=5_000, n_intervals=0),

        # ---------------------------
        # Sticky Top Navbar
        # ---------------------------
        html.Div(
            style={
                "width": "100%",
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "padding": "12px 24px",
                "backgroundColor": "rgba(15,23,42,0.85)",
                "color": "white",
                "position": "sticky",
                "top": "0",
                "zIndex": "1000",
            },
            children=[
                html.Div(
                    [
                        html.H2("Intelligent Digital Mentors", style={"margin": "0", "fontSize": "1.5rem"}),
                        html.P(
                            "Adaptive Learning Systems Dashboard (enhanced version)",
                            style={"margin": "0", "fontSize": "0.75rem", "opacity": "0.6"},
                        ),
                    ]
                ),
                html.Div(
                    style={"display": "flex", "gap": "12px", "alignItems": "center"},
                    children=[
                        # Secure field for storing an API key client-side only
                        dcc.Input(
                            id="input-api-key",
                            type="password",
                            placeholder="Enter your OpenAI API Key...",
                            style={
                                "width": "260px",
                                "padding": "6px 8px",
                                "borderRadius": "9999px",
                                "border": "1px solid rgba(255,255,255,0.2)",
                                "backgroundColor": "rgba(255,255,255,0.05)",
                                "color": "white",
                            },
                            value=DEFAULT_OPENAI_API_KEY,
                        ),
                        html.Button(
                            "Save API Key",
                            id="btn-save-api",
                            n_clicks=0,
                            style={
                                "background": "linear-gradient(120deg, #38bdf8, #6366f1)",
                                "border": "none",
                                "padding": "6px 14px",
                                "borderRadius": "9999px",
                                "color": "white",
                                "cursor": "pointer",
                                "fontWeight": "600",
                            },
                        ),
                    ],
                ),
            ],
        ),

        # ---------------------------
        # Main content area
        # ---------------------------
        html.Div(
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "16px",
                "padding": "16px",
            },
            children=[
                # ---------------------------------------
                # LEFT PANEL: Conversational Digital Mentor
                # ---------------------------------------
                html.Div(
                    style={
                        "flex": "1 1 360px",
                        "backgroundColor": "rgba(248,250,252,0.85)",
                        "borderRadius": "16px",
                        "padding": "16px",
                        "boxShadow": "0 10px 25px rgba(15,23,42,0.25)",
                        "minHeight": "520px",
                        "display": "flex",
                        "flexDirection": "column",
                        "maxHeight": "580px",
                    },
                    children=[
                        html.H3("Digital Mentor (ChatGPT)", style={"marginBottom": "8px", "color": "#0f172a"}),
                        html.P(
                            "The mentor adapts to the autonomy level and includes the real context of the selected student.",
                            style={"fontSize": "0.78rem", "color": "#475569"},
                        ),
                        # Container that renders the chat history
                        html.Div(
                            id="chat-container",
                            style={
                                "flex": "1",
                                "backgroundColor": "white",
                                "borderRadius": "12px",
                                "border": "1px solid #e2e8f0",
                                "padding": "10px",
                                "overflowY": "auto",
                                "maxHeight": "360px",
                            },
                        ),
                        # Message composer + send button
                        html.Div(
                            style={"display": "flex", "gap": "6px", "marginTop": "10px"},
                            children=[
                                dcc.Textarea(
                                    id="input-message",
                                    placeholder="Type your question…",
                                    style={
                                        "flex": "1",
                                        "minHeight": "60px",
                                        "borderRadius": "12px",
                                        "border": "1px solid #cbd5f5",
                                        "padding": "6px",
                                        "resize": "vertical",
                                    },
                                    value="",
                                ),
                                html.Button(
                                    "Send",
                                    id="btn-send",
                                    n_clicks=0,
                                    style={
                                        "background": "#0f172a",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "12px",
                                        "padding": "8px 14px",
                                        "cursor": "pointer",
                                        "alignSelf": "flex-end",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),

                # ---------------------------------------
                # RIGHT PANEL: Analytics and Controls
                # ---------------------------------------
                html.Div(
                    style={
                        "flex": "2 1 520px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "16px",
                    },
                    children=[
                        # ---- Header cards + selectors
                        html.Div(
                            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "alignItems": "stretch"},
                            children=[
                                # Average autonomy card
                                html.Div(
                                    style={
                                        "flex": "1 1 160px",
                                        "background": "linear-gradient(120deg, #38bdf8, #0ea5e9)",
                                        "color": "white",
                                        "borderRadius": "16px",
                                        "padding": "14px",
                                    },
                                    id="card-autonomy-average",
                                ),
                                # Average progress card
                                html.Div(
                                    style={
                                        "flex": "1 1 160px",
                                        "background": "linear-gradient(120deg, #6366f1, #8b5cf6)",
                                        "color": "white",
                                        "borderRadius": "16px",
                                        "padding": "14px",
                                    },
                                    id="card-progress-average",
                                ),
                                # Focused student selector
                                html.Div(
                                    style={
                                        "flex": "1 1 200px",
                                        "background": "white",
                                        "borderRadius": "16px",
                                        "padding": "10px",
                                        "boxShadow": "0 10px 25px rgba(15,23,42,0.12)",
                                    },
                                    children=[
                                        html.P("Focused student", style={"marginBottom": "2px", "fontSize": "0.75rem"}),
                                        dcc.Dropdown(
                                            id="dropdown-student",
                                            options=[
                                                {"label": s["name"], "value": s["id"]} for s in INITIAL_STUDENTS
                                            ],
                                            value=1,
                                            clearable=False,
                                            style={"fontSize": "0.75rem"},
                                        ),
                                        html.Div(id="mini-student-panel", style={"marginTop": "6px", "fontSize": "0.7rem"}),
                                    ],
                                ),
                                # Filter by module / cohort
                                html.Div(
                                    style={
                                        "flex": "1 1 220px",
                                        "background": "white",
                                        "borderRadius": "16px",
                                        "padding": "10px",
                                        "boxShadow": "0 10px 25px rgba(15,23,42,0.12)",
                                    },
                                    children=[
                                        html.P("Filter by module", style={"marginBottom": "4px", "fontSize": "0.75rem"}),
                                        dcc.Dropdown(
                                            id="dropdown-modules",
                                            options=[
                                                {"label": m, "value": m} for m in sorted({s["module"] for s in INITIAL_STUDENTS})
                                            ],
                                            value=[],
                                            multi=True,
                                            placeholder="All modules",
                                            style={"fontSize": "0.75rem"},
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        # ---- Main scatter (Autonomy vs Progress) + History panel
                        html.Div(
                            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                            children=[
                                # Fixed scatter plot (0–100 progress vs 0–10 autonomy)
                                html.Div(
                                    style={
                                        "flex": "1 1 380px",
                                        "backgroundColor": "white",
                                        "borderRadius": "16px",
                                        "padding": "14px",
                                        "boxShadow": "0 10px 25px rgba(15,23,42,0.12)",
                                        "height": "340px",
                                        "overflow": "hidden",
                                    },
                                    children=[
                                        html.H4("Autonomy vs Progress per student", style={"marginBottom": "4px"}),
                                        dcc.Graph(id="graph-autonomy-progress", style={"height": "280px"}),
                                    ],
                                ),
                                # History panel (simple line with most recent autonomy events)
                                html.Div(
                                    style={
                                        "flex": "1 1 260px",
                                        "backgroundColor": "white",
                                        "borderRadius": "16px",
                                        "padding": "14px",
                                        "boxShadow": "0 10px 25px rgba(15,23,42,0.12)",
                                        "height": "340px",
                                        "overflow": "hidden",
                                    },
                                    children=[
                                        html.H4("Autonomy history (simulated)", style={"marginBottom": "6px"}),
                                        dcc.RadioItems(
                                            id="radio-hist-window",
                                            options=[
                                                {"label": "Last 5 events", "value": "5"},
                                                {"label": "Last 10 events", "value": "10"},
                                                {"label": "Last 15 events", "value": "15"},
                                            ],
                                            value="10",
                                            inline=True,
                                            style={"fontSize": "0.7rem"},
                                        ),
                                        dcc.Graph(id="graph-autonomy-history", style={"height": "250px"}),
                                    ],
                                ),
                            ],
                        ),

                        # ---- Adaptive recommendations + latest autonomy log
                        html.Div(
                            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                            children=[
                                # Adaptive recommendations panel driven by slider
                                html.Div(
                                    style={
                                        "flex": "1 1 300px",
                                        "backgroundColor": "white",
                                        "borderRadius": "16px",
                                        "padding": "14px",
                                        "boxShadow": "0 10px 25px rgba(15,23,42,0.12)",
                                    },
                                    children=[
                                        html.H4("Adaptive recommendations", style={"marginBottom": "6px"}),
                                        dcc.Slider(
                                            0,
                                            100,
                                            5,
                                            value=40,
                                            id="slider-autonomy",
                                            marks={i: f"{i}%" for i in range(0, 101, 20)},
                                        ),
                                        html.Div(id="list-recommendations", style={"marginTop": "10px"}),
                                    ],
                                ),
                                # Log of latest simulated autonomy events
                                html.Div(
                                    style={
                                        "flex": "1 1 200px",
                                        "backgroundColor": "white",
                                        "borderRadius": "16px",
                                        "padding": "14px",
                                        "boxShadow": "0 10px 25px rgba(15,23,42,0.12)",
                                        "maxHeight": "200px",
                                        "overflowY": "auto",
                                    },
                                    children=[
                                        html.H4("Latest autonomy events", style={"marginBottom": "4px"}),
                                        html.Div(id="panel-autonomy-log", style={"fontSize": "0.7rem"}),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # ---------------------------
        # Footer
        # ---------------------------
        html.Div(
            style={
                "textAlign": "center",
                "padding": "10px 0",
                "fontSize": "0.7rem",
                "color": "white",
                "opacity": "0.5",
            },
            children=["Intelligent Digital Mentors – Adaptive Learning Systems – Dash + Plotly + OpenAI – Enhanced version."],
        ),
    ],
)

# =========================================================
# 5) CALLBACKS (APPLICATION LOGIC)
# ---------------------------------------------------------
# This section wires user interactions to UI updates.
# Each @app.callback defines Inputs/States and a function
# that returns the Outputs to update parts of the UI.
# =========================================================

# 5.1) Store the API key (so the user does not retype it)
@app.callback(
    Output("store-api-key", "data"),
    Input("btn-save-api", "n_clicks"),
    State("input-api-key", "value"),
    prevent_initial_call=True,
)
def save_api_key(n_clicks, api_key_value):
    """
    Persists the API key client-side (in a Dash Store).
    This is NOT secure for server-side secrets; it is only
    to keep the example self-contained and easy to test.
    """
    return {"api_key": api_key_value or ""}


# 5.2) Update resource recommendations according to slider (autonomy %)
@app.callback(
    Output("list-recommendations", "children"),
    Input("slider-autonomy", "value"),
)
def update_recommendations(autonomy_pct):
    """
    Rule-based content recommender:
    - Maps the autonomy percentage to a discrete level.
    - Returns a stylized list of resource suggestions.
    """
    if autonomy_pct is None:
        autonomy_pct = 40

    if autonomy_pct < 25:
        level = "very_low"
    elif autonomy_pct < 50:
        level = "low"
    elif autonomy_pct < 75:
        level = "medium"
    else:
        level = "high"

    resources = RESOURCES_BY_LEVEL.get(level, [])
    return [
        html.Div(
            r,
            style={
                "backgroundColor": "#0f172a0f",
                "padding": "6px 8px",
                "borderRadius": "10px",
                "marginBottom": "4px",
                "border": "1px solid rgba(15,23,42,0.05)",
            },
        )
        for r in resources
    ]


# 5.3) Chat with OpenAI (now including the context of the selected student)
@app.callback(
    Output("store-conversation", "data"),
    Input("btn-send", "n_clicks"),
    State("input-message", "value"),
    State("store-conversation", "data"),
    State("store-api-key", "data"),
    State("slider-autonomy", "value"),
    State("dropdown-student", "value"),
    State("store-students", "data"),
    State("list-recommendations", "children"),
    prevent_initial_call=True,
)
def converse_with_mentor(
    n_clicks,
    user_message,
    current_conversation,
    api_key_data,
    autonomy_pct,
    student_id,
    students_data,
    current_resources,
):
    """
    Builds a short, contextualized conversation and queries OpenAI:
    - Appends the new user message to local conversation history.
    - Crafts a system instruction according to the autonomy slider.
    - Injects selected student's module, progress, and autonomy.
    - Serializes current recommended resources (from the UI).
    - Calls safe_openai_call and appends the assistant response.
    """
    if current_conversation is None:
        current_conversation = []

    if not user_message or user_message.strip() == "":
        return current_conversation

    # Add user message to local history (simple structure)
    current_conversation.append(
        {
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat(),
        }
    )

    api_key = api_key_data.get("api_key", "") if api_key_data else ""
    if autonomy_pct is None:
        autonomy_pct = 40

    # Collect selected student's data
    selected_student = None
    if students_data:
        for st in students_data:
            if st["id"] == student_id:
                selected_student = st
                break

    # Choose the system instruction according to autonomy range
    if autonomy_pct < 25:
        base_instruction = (
            "You are a highly empathetic digital mentor for a learner with very low autonomy. "
            "Explain with simple examples, concrete steps, and explicit verification. "
            "Use academic Spanish and third-person tone."
        )
    elif autonomy_pct < 50:
        base_instruction = (
            "You are a digital mentor for a learner with low autonomy. "
            "Propose micro-activities and reinforcement. "
            "Use academic Spanish and third-person tone."
        )
    elif autonomy_pct < 75:
        base_instruction = (
            "You are a digital mentor for a learner with medium autonomy. "
            "Incorporate metacognition and application activities. "
            "Use academic Spanish and third-person tone."
        )
    else:
        base_instruction = (
            "You are a digital mentor for a learner with high autonomy. "
            "Pose open-ended challenges, advanced readings, and deepening routes. "
            "Use academic Spanish and third-person tone."
        )

    # Integrate the selected student's context into the prompt
    module_txt = selected_student["module"] if selected_student else "unidentified module"
    progress_txt = f"{round(selected_student['progress']*100,1)}%" if selected_student else "unknown"
    autonomy_txt = f"{round(selected_student['autonomy']*100,1)}%" if selected_student else f"{autonomy_pct}%"

    # Convert the current (UI) recommendations (Dash components) into plain text
    resources_text = []
    if current_resources:
        for r in current_resources:
            # Dash components arrive as dict-like structures with "props"
            if isinstance(r, dict) and "props" in r and "children" in r["props"]:
                resources_text.append(str(r["props"]["children"]))
            else:
                resources_text.append(str(r))
    resources_text = ", ".join(resources_text) if resources_text else "no suggested resources"

    # Final system instruction for this turn
    system_instruction = (
        base_instruction
        + f" The current learner works on: {module_txt}. "
        + f"Approximate progress: {progress_txt}; observed autonomy: {autonomy_txt}. "
        + f"Suggested resources by the adaptive system are: {resources_text}. "
        + "Integrate these resources in your explanation or mention them as next steps."
    )

    # Build a compact conversation history (last few turns)
    history = [{"role": "system", "content": system_instruction}]
    for m in current_conversation[-6:]:
        if m["role"] == "user":
            history.append({"role": "user", "content": m["content"]})
        else:
            history.append({"role": "assistant", "content": m["content"]})

    # Query OpenAI safely
    response = safe_openai_call(api_key, system_instruction, history)

    # Append assistant reply to local history
    current_conversation.append(
        {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat(),
        }
    )
    return current_conversation


# 5.4) Render the chat container from the stored conversation
@app.callback(
    Output("chat-container", "children"),
    Input("store-conversation", "data"),
)
def render_chat(conversation):
    """
    Displays the conversation as two types of message bubbles:
    - Right-aligned dark bubbles for user messages.
    - Left-aligned light bubbles for assistant responses.
    """
    if not conversation:
        return html.Div("No interaction yet.", style={"color": "#94a3b8"})
    elements = []
    for msg in conversation:
        if msg["role"] == "user":
            elements.append(
                html.Div(
                    msg["content"],
                    style={
                        "textAlign": "right",
                        "marginBottom": "6px",
                        "backgroundColor": "#0f172a",
                        "color": "white",
                        "padding": "6px 10px",
                        "borderRadius": "12px",
                        "maxWidth": "90%",
                        "marginLeft": "auto",
                    },
                )
            )
        else:
            elements.append(
                html.Div(
                    msg["content"],
                    style={
                        "textAlign": "left",
                        "marginBottom": "6px",
                        "backgroundColor": "#e2e8f0",
                        "color": "#0f172a",
                        "padding": "6px 10px",
                        "borderRadius": "12px",
                        "maxWidth": "90%",
                    },
                )
            )
    return elements


# 5.5) Interval-based simulation of student updates (progress & autonomy)
@app.callback(
    Output("store-students", "data"),
    Output("store-autonomy-log", "data"),
    Input("interval-update", "n_intervals"),
    State("store-students", "data"),
    State("store-autonomy-log", "data"),
)
def simulate_student_updates(n_intervals, students, autonomy_log):
    """
    Simulates non-linear learning dynamics:
    - Randomly picks a student and increases progress/autonomy
      by small random increments (bounded by [0,1]).
    - Appends a compact event to the autonomy_log (latest first).
    - Trims the log to keep only the last ~30 events for UI.
    """
    if students is None:
        students = INITIAL_STUDENTS
    if autonomy_log is None:
        autonomy_log = []

    idx = random.randint(0, len(students) - 1)
    st = students[idx]

    new_progress = min(st["progress"] + random.uniform(0.0, 0.03), 1.0)
    new_autonomy = min(st["autonomy"] + random.uniform(0.0, 0.015), 1.0)

    students[idx] = {
        **st,
        "progress": new_progress,
        "autonomy": new_autonomy,
    }

    autonomy_log.insert(
        0,
        {
            "name": st["name"],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "autonomy": round(new_autonomy * 100, 1),
        },
    )
    autonomy_log = autonomy_log[:30]  # Keep last 30 events

    return students, autonomy_log


# 5.6) Update the main scatter graph with optional module filtering and highlighting
@app.callback(
    Output("graph-autonomy-progress", "figure"),
    Input("store-students", "data"),
    Input("dropdown-student", "value"),
    Input("dropdown-modules", "value"),
)
def update_main_scatter(students, selected_id, filtered_modules):
    """
    Builds a scatter plot:
    - X axis: Progress (%) in [0,100].
    - Y axis: Autonomy (scaled to [0,10]).
    - Color: Persistently assigned by module.
    - Size: Larger marker for the currently selected student.
    """
    if students is None:
        students = INITIAL_STUDENTS

    # Filter by selected modules
    if filtered_modules:
        filtered = [s for s in students if s["module"] in filtered_modules]
    else:
        filtered = students

    # Prepare vectors
    x_vals = [s["progress"] * 100 for s in filtered]
    y_vals = [(s["autonomy"] * 100) / 10 for s in filtered]  # 0–10 scale
    names = [s["name"] for s in filtered]
    modules = [s["module"] for s in filtered]

    # Consistent colors per module
    colors = [MOD_COLOR_MAP.get(m, "#94a3b8") for m in modules]

    # Emphasize the selected student
    sizes = [28 if s["id"] == selected_id else 18 for s in filtered]

    fig = px.scatter(
        x=x_vals,
        y=y_vals,
        hover_name=names,
    )
    fig.update_traces(
        marker=dict(color=colors, size=sizes, line=dict(width=1, color="#0f172a"))
    )
    fig.update_yaxes(range=[0, 10], title="Autonomy (0–10)")
    fig.update_xaxes(range=[0, 100], title="Progress (%)")
    fig.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=20, r=10, t=10, b=10),
        transition_duration=400,
        showlegend=False,
    )
    return fig


# 5.7) Update header cards (averages)
@app.callback(
    Output("card-autonomy-average", "children"),
    Output("card-progress-average", "children"),
    Input("store-students", "data"),
)
def update_header_cards(students):
    """
    Computes cohort-level averages:
    - Mean autonomy as percentage.
    - Mean progress as percentage.
    Renders concise texts inside the two header cards.
    """
    if students is None or len(students) == 0:
        return "No data", "No data"

    mean_autonomy = round(sum([s["autonomy"] for s in students]) / len(students) * 100, 1)
    mean_progress = round(sum([s["progress"] for s in students]) / len(students) * 100, 1)

    card1 = [
        html.P("Average autonomy", style={"margin": "0", "opacity": "0.7"}),
        html.H2(f"{mean_autonomy}%", style={"margin": "0"}),
        html.P(f"Students: {len(students)}", style={"marginBottom": "0", "fontSize": "0.7rem"}),
    ]
    card2 = [
        html.P("Average progress", style={"margin": "0", "opacity": "0.7"}),
        html.H2(f"{mean_progress}%", style={"margin": "0"}),
        html.P("Active modules: 4", style={"marginBottom": "0", "fontSize": "0.7rem"}),
    ]
    return card1, card2


# 5.8) Mini panel for currently focused student
@app.callback(
    Output("mini-student-panel", "children"),
    Input("dropdown-student", "value"),
    State("store-students", "data"),
)
def show_student_info(student_id, students):
    """
    Renders a compact summary of the selected student:
    module, progress %, autonomy %.
    """
    if students is None:
        students = INITIAL_STUDENTS
    st = next((s for s in students if s["id"] == student_id), None)
    if not st:
        return "Student not found."
    return html.Div(
        [
            html.Div(f"Module: {st['module']}"),
            html.Div(f"Progress: {round(st['progress']*100,1)}%"),
            html.Div(f"Autonomy: {round(st['autonomy']*100,1)}%"),
        ]
    )


# 5.9) Render the latest autonomy events log
@app.callback(
    Output("panel-autonomy-log", "children"),
    Input("store-autonomy-log", "data"),
)
def render_autonomy_log(log):
    """
    Displays the most recent autonomy updates as a simple list:
    timestamp — student name: autonomy%
    """
    if not log:
        return html.Div("No events yet.")
    return [
        html.Div(
            f"{item['timestamp']} — {item['name']}: {item['autonomy']}%",
            style={"padding": "2px 0", "borderBottom": "1px solid #e2e8f0"},
        )
        for item in log
    ]


# 5.10) Autonomy history chart for the selected student (or fallback to all)
@app.callback(
    Output("graph-autonomy-history", "figure"),
    Input("store-autonomy-log", "data"),
    Input("radio-hist-window", "value"),
    Input("dropdown-student", "value"),
)
def update_autonomy_history(log, window, selected_student_id):
    """
    Builds a simple line chart with the last N autonomy events.
    Notes:
    - The original heuristic attempted to match the student's id
      inside the event 'name'. Since names like "Student A" do not
      end with the numeric id, the code includes a fallback that uses
      the full log if no specific match is found.
    - 'window' controls how many recent events to display (5/10/15).
    """
    # Empty figure if there is no data yet
    if not log:
        fig = px.line()
        fig.update_layout(
            template="plotly_white",
            height=250,
            margin=dict(l=20, r=10, t=10, b=10),
            xaxis_title="Time (events)",
            yaxis_title="Autonomy (%)",
        )
        return fig

    # Parse window size safely
    try:
        window = int(window)
    except Exception:
        window = 10

    # Attempt to filter by selected student using a naive heuristic.
    # Since names do not encode the id, this typically yields no match.
    # The fallback is to use all events.
    filtered_log = [e for e in log if e["name"].endswith(f"{selected_student_id}")]
    if not filtered_log:
        filtered_log = log

    # Keep only the 'window' most recent events and reverse for temporal order
    data = filtered_log[:window]
    data = list(reversed(data))

    fig = px.line(
        x=list(range(1, len(data) + 1)),
        y=[d["autonomy"] for d in data],
    )
    fig.update_layout(
        template="plotly_white",
        height=250,
        margin=dict(l=20, r=10, t=10, b=10),
        xaxis_title="Recent events",
        yaxis_title="Autonomy (%)",
        yaxis=dict(range=[0, 110]),
    )
    return fig


# =========================================================
# 6) MAIN ENTRY POINT
# ---------------------------------------------------------
# Standard Dash launcher. In local Windows environments, it
# serves on 127.0.0.1:8050 with debug enabled.
# =========================================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=True)

# http://127.0.0.1:8050/
