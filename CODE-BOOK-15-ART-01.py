
"""
Research pipeline for chapter sections 5.1, 5.2, and 5.3
Persistent Homology for Learning Trajectories with Cycles, Transitions, and Ruptures

This script is intentionally written as a complete, research-oriented pipeline aligned
with the chapter's objective: to show how persistent homology and related topological
tools can reveal cycles, transitions, ruptures, and actionable educational signals in
learning trajectories.

What this script does
---------------------
1. Generates a synthetic educational dataset with:
   - temporally ordered event logs,
   - weekly learner states,
   - concept mastery traces,
   - three interpretable learner archetypes:
       * cyclic regulation,
       * pedagogical transition,
       * structural rupture.

2. Builds a preprocessing pipeline consistent with the chapter:
   - event aggregation,
   - weekly state construction,
   - robust scaling,
   - temporal smoothing,
   - trajectory point clouds.

3. Produces outputs for chapter section 5.1 using:
   - GUDHI,
   - Ripser.py,
   - Persim.
   The section generates:
   - persistence diagrams,
   - topological barcodes,
   - Betti curves,
   - cycle summary tables.

4. Produces outputs for chapter section 5.2 using:
   - Ripser.py / Scikit-TDA,
   - manual Betti-curve and persistence-entropy replacements,
   - a lightweight topology-aware deep autoencoder.
   The section generates:
   - latent-space topology evolution across epochs,
   - before/after topology comparisons,
   - model-comparison tables.

5. Produces outputs for chapter section 5.3 using:
   - KeplerMapper,
   - TopoNetX.
   The section generates:
   - a topological knowledge map,
   - a learning graph derived from a simplicial complex,
   - actionable recommendation tables.

Important note on runtime
-------------------------
The code is intentionally designed to remain relatively fast. The synthetic dataset is
moderate in size, persistent homology is computed in low dimension (H0 and H1), and
the topological regularization in the deep model uses a lightweight, differentiable
neighborhood-preservation surrogate during training, while persistent homology is used
for periodic auditing of the latent space.

This makes the script appropriate for chapter-quality demonstrations: it generates
meaningful figures and tables without requiring large-scale compute.

How to use
----------
python chapter_topology_pipeline_complete.py --output-root ./chapter_outputs

Suggested dependencies
----------------------
pip install numpy pandas matplotlib networkx scikit-learn scipy torch gudhi ripser persim kmapper toponetx

The script is not executed automatically on import.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, silhouette_score, brier_score_loss
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from scipy.stats import entropy as scipy_entropy

# ---------------------------------------------------------------------
# Optional third-party dependencies. The script checks them explicitly
# inside main() so that the file can still be opened and inspected even
# if some packages are not installed yet.
# ---------------------------------------------------------------------
MISSING_PACKAGES = []

try:
    import gudhi
except Exception:
    gudhi = None
    MISSING_PACKAGES.append("gudhi")

try:
    from ripser import ripser, Rips
except Exception:
    ripser = None
    Rips = None
    MISSING_PACKAGES.append("ripser")

try:
    from persim import plot_diagrams, bottleneck, wasserstein
except Exception:
    plot_diagrams = None
    bottleneck = None
    wasserstein = None
    MISSING_PACKAGES.append("persim")

# giotto-tda is intentionally not required in this version of the script.
# Section 5.2 below reproduces the needed Vietoris–Rips, Betti-curve, and
# persistence-entropy functionality with Ripser and manual vectorizations.

try:
    import kmapper as km
except Exception:
    km = None
    MISSING_PACKAGES.append("kmapper")

try:
    import toponetx as tnx
except Exception:
    tnx = None
    MISSING_PACKAGES.append("toponetx")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None
    MISSING_PACKAGES.append("torch")

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass
class PipelineConfig:
    seed: int = 42
    n_learners_per_group: int = 24
    n_weeks: int = 12
    n_concepts: int = 8
    smoothing_alpha: float = 0.65
    min_weekly_events: int = 4
    max_ph_dimension: int = 1
    autoencoder_epochs: int = 18
    autoencoder_batch_size: int = 64
    autoencoder_lr: float = 1e-3
    topo_weight: float = 0.30
    audit_every: int = 2
    latent_dim: int = 2
    mapper_n_cubes: int = 10
    mapper_overlap: float = 0.25
    output_root: str = "chapter_outputs"


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------
def check_dependencies() -> None:
    """Raise a user-friendly error if required packages are missing."""
    if MISSING_PACKAGES:
        missing = sorted(set(MISSING_PACKAGES))
        raise ImportError(
            "The following packages are required by this script but are not installed: "
            + ", ".join(missing)
            + ".\nInstall them first, for example:\n"
            + "pip install numpy pandas matplotlib networkx scikit-learn scipy torch gudhi "
              "ripser persim kmapper toponetx"
        )


def set_global_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def clip01(x: np.ndarray | float) -> np.ndarray | float:
    """Clip a numeric value or array to the [0, 1] range."""
    return np.clip(x, 0.0, 1.0)


def ensure_output_tree(root: str) -> Dict[str, Path]:
    """
    Create the exact directory tree used by the chapter outputs.

    The structure mirrors the article sections so that figures and tables can be
    placed directly into sections 5.1, 5.2, and 5.3.
    """
    root_path = Path(root)
    paths = {
        "root": root_path,
        "shared": root_path / "shared",
        "shared_data": root_path / "shared" / "data",
        "shared_intermediate": root_path / "shared" / "intermediate",
        "s51": root_path / "5_1_detection_of_cycles",
        "s51_fig": root_path / "5_1_detection_of_cycles" / "figures",
        "s51_tab": root_path / "5_1_detection_of_cycles" / "tables",
        "s52": root_path / "5_2_topological_regularization",
        "s52_fig": root_path / "5_2_topological_regularization" / "figures",
        "s52_tab": root_path / "5_2_topological_regularization" / "tables",
        "s53": root_path / "5_3_topological_recommendations",
        "s53_fig": root_path / "5_3_topological_recommendations" / "figures",
        "s53_tab": root_path / "5_3_topological_recommendations" / "tables",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV with stable formatting."""
    df.to_csv(path, index=False)


def save_png(path: Path, dpi: int = 220) -> None:
    """Standardized figure export."""
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def minmax_series(values: pd.Series) -> pd.Series:
    """Scale a pandas Series into [0, 1] while protecting against division by zero."""
    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - vmin) / (vmax - vmin)


# ---------------------------------------------------------------------
# Synthetic educational data generation
# ---------------------------------------------------------------------
def get_concepts_and_prerequisites() -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Define a concept set coherent with the chapter's educational logic.

    The concepts are written so that the later recommendation system can produce
    interpretable outputs directly aligned with the chapter.
    """
    concepts = [
        "Event Logs",
        "Temporal Windows",
        "State Encoding",
        "Similarity Design",
        "Filtration",
        "Homology",
        "Persistence Diagrams",
        "Educational Signals",
    ]
    prereq = {
        "Event Logs": [],
        "Temporal Windows": ["Event Logs"],
        "State Encoding": ["Event Logs", "Temporal Windows"],
        "Similarity Design": ["State Encoding"],
        "Filtration": ["State Encoding", "Similarity Design"],
        "Homology": ["Filtration"],
        "Persistence Diagrams": ["Homology"],
        "Educational Signals": ["Persistence Diagrams"],
    }
    return concepts, prereq


def group_offset(group: str) -> int:
    """Small helper for creating stable numeric group identifiers."""
    return {"cycle": 0, "transition": 1, "rupture": 2}[group]


def simulate_hidden_weekly_profiles(config: PipelineConfig) -> pd.DataFrame:
    """
    Create the hidden weekly learner profiles that drive the synthetic event generator.

    The hidden variables are not the final outputs used in the analysis. Instead, they
    are latent pedagogical profiles used to generate realistic event logs and mastery
    patterns. This keeps the workflow consistent with the article's architecture: event
    traces come first, then state construction, then topology.
    """
    rng = np.random.default_rng(config.seed)
    concepts, prereq = get_concepts_and_prerequisites()
    all_rows: List[Dict[str, float | int | str]] = []

    groups = ["cycle", "transition", "rupture"]

    for group in groups:
        for learner_idx in range(config.n_learners_per_group):
            learner_id = f"{group[:3].upper()}_{learner_idx:03d}"
            mastery = {c: 0.05 + 0.03 * rng.random() for c in concepts}
            phase_offset = rng.uniform(0, 2 * np.pi)

            weekly_rows: List[Dict[str, float | int | str]] = []

            for week in range(1, config.n_weeks + 1):
                theta = 2 * np.pi * (week - 1) / config.n_weeks + phase_offset

                # ---------------------------------------------------------
                # Behavior archetypes
                # ---------------------------------------------------------
                if group == "cycle":
                    # Recurrent study-feedback-practice behavior:
                    # the point cloud should naturally form loop-like patterns.
                    engagement = 0.70 + 0.10 * np.sin(theta) + rng.normal(0, 0.03)
                    performance = 0.58 + 0.18 * (week / config.n_weeks) + 0.06 * np.cos(2 * theta) + rng.normal(0, 0.03)
                    help_seeking = 0.48 + 0.15 * np.sin(theta + np.pi / 4) + rng.normal(0, 0.03)
                    review_ratio = 0.52 + 0.18 * np.cos(theta - np.pi / 6) + rng.normal(0, 0.03)
                    regularity = 0.72 + 0.08 * np.sin(theta + np.pi / 3) + rng.normal(0, 0.02)
                    topic_diversity = 0.50 + 0.10 * np.cos(theta + np.pi / 7) + rng.normal(0, 0.02)
                    latent_x = np.cos(theta) + rng.normal(0, 0.08)
                    latent_y = np.sin(theta) + rng.normal(0, 0.08)
                    active_concepts = [
                        concepts[(week - 1) % 4],
                        concepts[week % 4],
                        concepts[min(week // 2, len(concepts) - 1)],
                    ]
                    assessment_event = int(week in [4, 8, 12])
                    shock_event = 0

                elif group == "transition":
                    # A pedagogical shift around the middle of the course:
                    # first broad exploration, then more focused consolidation.
                    progress = sigmoid((week - 6.0) / 1.2)
                    engagement = 0.46 + 0.28 * progress + rng.normal(0, 0.03)
                    performance = 0.42 + 0.34 * progress + rng.normal(0, 0.03)
                    help_seeking = 0.30 + 0.25 * np.exp(-((week - 6.0) ** 2) / 3.0) + rng.normal(0, 0.02)
                    review_ratio = 0.34 + 0.18 * progress + rng.normal(0, 0.02)
                    regularity = 0.42 + 0.30 * progress + rng.normal(0, 0.02)
                    topic_diversity = 0.72 - 0.28 * progress + rng.normal(0, 0.02)
                    latent_x = -1.2 + 2.2 * progress + rng.normal(0, 0.10)
                    latent_y = -0.8 + 1.6 * progress + rng.normal(0, 0.10)
                    if week <= 5:
                        active_concepts = ["Event Logs", "Temporal Windows", "State Encoding"]
                    elif week <= 8:
                        active_concepts = ["State Encoding", "Similarity Design", "Filtration", "Homology"]
                    else:
                        active_concepts = ["Homology", "Persistence Diagrams", "Educational Signals"]
                    assessment_event = int(week in [6, 7])
                    shock_event = 0

                else:
                    # Early coherence followed by fragmentation and decline.
                    # This should support rupture-related metrics later.
                    if week <= 6:
                        engagement = 0.66 + 0.04 * np.sin(theta) + rng.normal(0, 0.03)
                        performance = 0.55 + 0.12 * (week / 6.0) + rng.normal(0, 0.03)
                        help_seeking = 0.42 + 0.06 * np.cos(theta) + rng.normal(0, 0.02)
                        review_ratio = 0.39 + 0.06 * np.sin(theta) + rng.normal(0, 0.02)
                        regularity = 0.66 + 0.05 * np.cos(theta) + rng.normal(0, 0.02)
                        topic_diversity = 0.44 + 0.06 * np.sin(theta) + rng.normal(0, 0.02)
                        latent_x = 0.25 + 0.10 * week + rng.normal(0, 0.06)
                        latent_y = 0.90 - 0.07 * week + rng.normal(0, 0.06)
                    else:
                        engagement = 0.36 - 0.04 * (week - 6.0) + rng.normal(0, 0.04)
                        performance = 0.43 - 0.07 * (week - 6.0) + rng.normal(0, 0.04)
                        help_seeking = 0.20 + 0.18 * rng.random()
                        review_ratio = 0.18 + 0.20 * rng.random()
                        regularity = 0.20 + 0.18 * rng.random()
                        topic_diversity = 0.18 + 0.25 * rng.random()
                        latent_x = rng.normal(0.0, 0.90)
                        latent_y = rng.normal(0.0, 0.90)
                    active_concepts = ["Event Logs", "Temporal Windows", "State Encoding"] if week <= 7 else ["Event Logs", "Temporal Windows"]
                    assessment_event = int(week == 6)
                    shock_event = int(week >= 7)

                engagement = float(clip01(engagement))
                performance = float(clip01(performance))
                help_seeking = float(clip01(help_seeking))
                review_ratio = float(clip01(review_ratio))
                regularity = float(clip01(regularity))
                topic_diversity = float(clip01(topic_diversity))
                target_event_count = int(np.clip(np.round(6 + 14 * engagement + rng.normal(0, 1.2)), 4, 20))

                # ---------------------------------------------------------
                # Mastery evolution over the concept chain
                # ---------------------------------------------------------
                for concept in concepts:
                    prereq_support = 1.0
                    if prereq[concept]:
                        prereq_support = float(np.mean([mastery[p] for p in prereq[concept]]))

                    is_active = 1.0 if concept in active_concepts else 0.25
                    if group == "cycle":
                        gain = (0.05 + 0.05 * review_ratio) * engagement * (0.55 + 0.45 * prereq_support) * is_active
                        forgetting = 0.01 * (1.0 - review_ratio) * mastery[concept]
                    elif group == "transition":
                        base = 0.035 if week <= 5 else 0.085
                        gain = base * engagement * (0.45 + 0.55 * prereq_support) * is_active
                        forgetting = 0.008 * (1.0 - regularity) * mastery[concept]
                    else:
                        if week <= 6:
                            gain = 0.055 * engagement * (0.40 + 0.60 * prereq_support) * is_active
                            forgetting = 0.010 * (1.0 - review_ratio) * mastery[concept]
                        else:
                            gain = 0.020 * engagement * (0.25 + 0.75 * prereq_support) * is_active
                            forgetting = 0.050 * (1.0 - regularity) * mastery[concept]

                    mastery[concept] = float(clip01(mastery[concept] + gain - forgetting + rng.normal(0, 0.010)))

                week_row = {
                    "learner_id": learner_id,
                    "trajectory_group": group,
                    "week": week,
                    "engagement_hidden": engagement,
                    "performance_hidden": performance,
                    "help_seeking_hidden": help_seeking,
                    "review_ratio_hidden": review_ratio,
                    "regularity_hidden": regularity,
                    "topic_diversity_hidden": topic_diversity,
                    "latent_x_hidden": latent_x,
                    "latent_y_hidden": latent_y,
                    "target_event_count": target_event_count,
                    "assessment_event": assessment_event,
                    "shock_event": shock_event,
                }
                for concept in concepts:
                    week_row[f"mastery_{concept}"] = mastery[concept]

                weekly_rows.append(week_row)

            # Final risk is learner-level and repeated across weeks.
            final_week = weekly_rows[-1]
            final_mastery = np.mean([final_week[f"mastery_{c}"] for c in concepts])
            final_perf = float(final_week["performance_hidden"])
            final_risk = int((group == "rupture") or (final_mastery < 0.58) or (final_perf < 0.50))

            for row in weekly_rows:
                row["final_risk"] = final_risk
                all_rows.append(row)

    hidden_df = pd.DataFrame(all_rows)
    return hidden_df


def choose_topic_weights(
    concept_names: Sequence[str],
    mastery_values: Dict[str, float],
    active_concepts: Sequence[str],
    rng: np.random.Generator
) -> np.ndarray:
    """
    Build topic-sampling weights for event generation.

    Topics with lower mastery and topics marked as active in the hidden profile
    receive more probability mass, which makes the event log pedagogically coherent.
    """
    weights = []
    active_set = set(active_concepts)
    for concept in concept_names:
        mastery_gap = 1.0 - mastery_values[concept]
        active_boost = 1.6 if concept in active_set else 1.0
        weights.append((0.20 + mastery_gap) * active_boost)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return weights


def infer_active_concepts_from_row(row: pd.Series, concept_names: Sequence[str]) -> List[str]:
    """
    Recover the active concept subset from the hidden weekly profile.

    Because the hidden generator already encodes mastery evolution, here the active
    concepts are approximated as the concepts with the largest weekly mastery gaps.
    """
    mastery_gap = {c: 1.0 - float(row[f"mastery_{c}"]) for c in concept_names}
    ordered = sorted(mastery_gap.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ordered[:3]]


def generate_event_logs(hidden_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Generate event logs from hidden weekly profiles.

    The event logs are the raw traces that later get aggregated into weekly learner states,
    thereby matching the article's methodological logic.
    """
    rng = np.random.default_rng(config.seed + 11)
    concept_names, _ = get_concepts_and_prerequisites()

    event_rows: List[Dict[str, float | int | str]] = []
    event_id = 0

    for _, row in hidden_df.iterrows():
        learner_id = row["learner_id"]
        group = row["trajectory_group"]
        week = int(row["week"])

        engagement = float(row["engagement_hidden"])
        performance = float(row["performance_hidden"])
        help_seeking = float(row["help_seeking_hidden"])
        review_ratio = float(row["review_ratio_hidden"])
        regularity = float(row["regularity_hidden"])
        n_events = int(row["target_event_count"])

        mastery_values = {c: float(row[f"mastery_{c}"]) for c in concept_names}
        active_concepts = infer_active_concepts_from_row(row, concept_names)

        # Day allocation becomes more uneven when regularity is low.
        concentration = 1.0 + 18.0 * regularity
        daily_probs = rng.dirichlet(np.repeat(concentration / 7.0, 7))

        # Event-type probabilities are group-sensitive and feature-sensitive.
        if group == "cycle":
            action_names = ["study", "feedback", "practice", "review", "quiz", "forum"]
            action_probs = np.array([
                0.24,
                0.15 + 0.10 * help_seeking,
                0.20,
                0.18 + 0.15 * review_ratio,
                0.15,
                0.08,
            ], dtype=float)
        elif group == "transition":
            action_names = ["study", "feedback", "practice", "review", "quiz", "forum"]
            action_probs = np.array([
                0.28,
                0.10 + 0.15 * help_seeking,
                0.18,
                0.12 + 0.10 * review_ratio,
                0.20,
                0.12,
            ], dtype=float)
        else:
            action_names = ["study", "feedback", "practice", "review", "quiz", "forum"]
            action_probs = np.array([
                0.26,
                0.08 + 0.08 * help_seeking,
                0.16,
                0.10 + 0.10 * review_ratio,
                0.16,
                0.24,
            ], dtype=float)

        action_probs = action_probs / action_probs.sum()
        topic_probs = choose_topic_weights(concept_names, mastery_values, active_concepts, rng)

        for _event in range(n_events):
            day_of_week = int(rng.choice(np.arange(1, 8), p=daily_probs))
            action = str(rng.choice(action_names, p=action_probs))
            topic = str(rng.choice(concept_names, p=topic_probs))

            duration_minutes = max(4.0, rng.normal(12 + 35 * engagement, 4.5))
            if action in {"quiz", "practice"}:
                correctness = float(clip01(rng.normal(performance, 0.10)))
            else:
                correctness = np.nan

            support_signal = float(clip01(rng.normal(help_seeking if action in {"feedback", "forum"} else 0.20, 0.08)))

            event_rows.append({
                "event_id": event_id,
                "learner_id": learner_id,
                "trajectory_group": group,
                "week": week,
                "day_of_week": day_of_week,
                "action": action,
                "topic": topic,
                "duration_minutes": float(duration_minutes),
                "correctness": correctness,
                "support_signal": support_signal,
            })
            event_id += 1

    events_df = pd.DataFrame(event_rows)
    return events_df


def aggregate_weekly_states(events_df: pd.DataFrame, hidden_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate event logs into weekly learner states.

    This function is the explicit bridge from raw event logs to temporally ordered states,
    exactly as required by the chapter.
    """
    concept_names, _ = get_concepts_and_prerequisites()

    grouped = events_df.groupby(["learner_id", "trajectory_group", "week"], as_index=False)

    summary_rows: List[Dict[str, float | int | str]] = []

    for (learner_id, group, week), chunk in events_df.groupby(["learner_id", "trajectory_group", "week"]):
        action_counts = Counter(chunk["action"])
        total_events = int(len(chunk))
        mean_duration = float(chunk["duration_minutes"].mean())
        quiz_practice = chunk.loc[chunk["action"].isin(["quiz", "practice"]), "correctness"].dropna()
        performance = float(quiz_practice.mean()) if len(quiz_practice) > 0 else 0.45

        # Regularity uses the variability of daily counts. Low variation means
        # temporally stable behavior; high variation means uneven or erratic use.
        day_counts = chunk.groupby("day_of_week").size().reindex(range(1, 8), fill_value=0).values.astype(float)
        mean_day = float(day_counts.mean())
        std_day = float(day_counts.std())
        cv_day = std_day / (mean_day + 1e-8)
        regularity = float(1.0 / (1.0 + cv_day))

        topic_diversity = float(chunk["topic"].nunique() / max(1, len(concept_names)))
        help_seeking = float((action_counts.get("feedback", 0) + action_counts.get("forum", 0)) / total_events)
        review_ratio = float(action_counts.get("review", 0) / total_events)

        # Engagement mixes amount and duration and remains normalized later.
        engagement_raw = float(total_events * mean_duration)

        row = {
            "learner_id": learner_id,
            "trajectory_group": group,
            "week": int(week),
            "event_count": total_events,
            "mean_duration_minutes": mean_duration,
            "engagement_raw": engagement_raw,
            "performance": performance,
            "help_seeking": help_seeking,
            "review_ratio": review_ratio,
            "regularity": regularity,
            "topic_diversity": topic_diversity,
            "study_ratio": float(action_counts.get("study", 0) / total_events),
            "practice_ratio": float(action_counts.get("practice", 0) / total_events),
            "quiz_ratio": float(action_counts.get("quiz", 0) / total_events),
            "forum_ratio": float(action_counts.get("forum", 0) / total_events),
        }
        summary_rows.append(row)

    weekly_df = pd.DataFrame(summary_rows)

    # Normalize engagement after aggregation so that it stays on a 0-1 scale.
    weekly_df["engagement"] = minmax_series(weekly_df["engagement_raw"])

    # Merge hidden concept mastery and learner-level outcomes.
    keep_cols = ["learner_id", "week", "assessment_event", "shock_event", "final_risk"]
    keep_cols += [c for c in hidden_df.columns if c.startswith("mastery_")]
    merge_df = hidden_df[keep_cols].copy()

    weekly_df = weekly_df.merge(merge_df, on=["learner_id", "week"], how="left")

    # Reorder a bit for readability.
    main_cols = [
        "learner_id", "trajectory_group", "week", "final_risk",
        "event_count", "engagement", "performance", "help_seeking",
        "review_ratio", "regularity", "topic_diversity",
        "study_ratio", "practice_ratio", "quiz_ratio", "forum_ratio",
        "assessment_event", "shock_event"
    ]
    mastery_cols = [c for c in weekly_df.columns if c.startswith("mastery_")]
    weekly_df = weekly_df[main_cols + mastery_cols]

    return weekly_df.sort_values(["learner_id", "week"]).reset_index(drop=True)


def preprocess_weekly_states(weekly_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Apply robust scaling and temporal smoothing to the weekly states.

    This keeps the preprocessing close to the chapter's proposed computational pipeline.
    """
    df = weekly_df.copy()
    behavior_cols = [
        "engagement", "performance", "help_seeking", "review_ratio",
        "regularity", "topic_diversity", "study_ratio", "practice_ratio",
        "quiz_ratio", "forum_ratio"
    ]

    scaler = RobustScaler()
    df[[f"{c}_scaled" for c in behavior_cols]] = scaler.fit_transform(df[behavior_cols])

    smoothed_rows = []
    smoothed_cols = [f"{c}_scaled" for c in behavior_cols]

    for learner_id, chunk in df.groupby("learner_id", sort=False):
        chunk = chunk.sort_values("week").copy()
        prev = None
        for idx in chunk.index:
            current = chunk.loc[idx, smoothed_cols].to_numpy(dtype=float)
            if prev is None:
                smooth = current.copy()
            else:
                smooth = config.smoothing_alpha * current + (1.0 - config.smoothing_alpha) * prev
            prev = smooth
            for col, value in zip(smoothed_cols, smooth):
                chunk.loc[idx, f"{col}_smooth"] = float(value)
        smoothed_rows.append(chunk)

    df = pd.concat(smoothed_rows, axis=0).sort_values(["learner_id", "week"]).reset_index(drop=True)

    # Activity threshold flag.
    df["keep_for_topology"] = (df["event_count"] >= config.min_weekly_events).astype(int)
    return df


# ---------------------------------------------------------------------
# Topological helper functions
# ---------------------------------------------------------------------
def get_behavior_point_cloud(df: pd.DataFrame, learner_id: str) -> np.ndarray:
    """
    Build a learner-level point cloud from smoothed behavioral states.

    The selected variables are intentionally aligned with cycles, transitions, and
    ruptures in learning behavior.
    """
    cols = [
        "engagement_scaled_smooth",
        "performance_scaled_smooth",
        "help_seeking_scaled_smooth",
        "review_ratio_scaled_smooth",
        "regularity_scaled_smooth",
        "topic_diversity_scaled_smooth",
    ]
    chunk = df[(df["learner_id"] == learner_id) & (df["keep_for_topology"] == 1)].sort_values("week")
    return chunk[cols].to_numpy(dtype=float)


def finite_h1(diagrams: Sequence[np.ndarray]) -> np.ndarray:
    """Extract the finite H1 pairs from Ripser output."""
    if len(diagrams) < 2:
        return np.empty((0, 2))
    h1 = np.asarray(diagrams[1], dtype=float)
    if h1.size == 0:
        return np.empty((0, 2))
    keep = np.isfinite(h1[:, 1])
    return h1[keep]


def total_persistence(h1: np.ndarray) -> float:
    """Total persistence in H1."""
    if h1.size == 0:
        return 0.0
    return float(np.sum(h1[:, 1] - h1[:, 0]))


def max_lifetime(h1: np.ndarray) -> float:
    """Largest H1 lifetime."""
    if h1.size == 0:
        return 0.0
    return float(np.max(h1[:, 1] - h1[:, 0]))


def short_lived_ratio(h1: np.ndarray, quantile: float = 0.35) -> float:
    """
    Ratio of short-lived H1 features.

    This is useful later for rupture-oriented summaries: a sudden shift toward
    very short-lived features often indicates structural instability rather than
    robust recurring organization.
    """
    if h1.size == 0:
        return 1.0
    lifetimes = h1[:, 1] - h1[:, 0]
    threshold = np.quantile(lifetimes, quantile)
    return float(np.mean(lifetimes <= threshold))


def h0_long_lived_count(diagrams: Sequence[np.ndarray], threshold: float = 0.35) -> int:
    """Count H0 features with death time above a threshold, excluding the infinite class."""
    h0 = np.asarray(diagrams[0], dtype=float)
    if h0.size == 0:
        return 0
    keep = np.isfinite(h0[:, 1]) & (h0[:, 1] >= threshold)
    return int(keep.sum())


def betti_curve_manual(h1: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Manual H1 Betti curve from a persistence diagram."""
    if h1.size == 0:
        return np.zeros_like(grid, dtype=float)
    return np.array([np.sum((h1[:, 0] <= t) & (h1[:, 1] > t)) for t in grid], dtype=float)


def gudhi_persistence_from_cloud(point_cloud: np.ndarray, max_dimension: int = 2):
    """
    Compute a GUDHI simplex tree and persistence pairs from a point cloud.

    GUDHI is used here because section 5.1 explicitly requests barcode-style
    educational visualizations.
    """
    if point_cloud.shape[0] < 3:
        return None, []

    pairwise = np.linalg.norm(point_cloud[:, None, :] - point_cloud[None, :, :], axis=2)
    max_edge = float(np.quantile(pairwise, 0.85))
    max_edge = max(max_edge, 0.25)

    rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=max_edge)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    diag = simplex_tree.persistence()
    return simplex_tree, diag


def h1_diagram_for_cloud(
    point_cloud: np.ndarray,
    metric: str = "euclidean",
    coeff: int = 2,
    max_edge_length: float = np.inf,
) -> np.ndarray:
    """
    Compute the finite H1 persistence diagram for a single point cloud with Ripser.

    This replaces the original giotto-tda VietorisRipsPersistence call used only to
    derive Betti curves and persistence entropy in section 5.2.
    """
    result = ripser(
        point_cloud,
        maxdim=1,
        metric=metric,
        coeff=coeff,
        thresh=max_edge_length,
    )["dgms"]
    return finite_h1(result)


def giotto_like_betti_sampling(h1: np.ndarray, n_bins: int = 100) -> np.ndarray:
    """
    Reproduce giotto-tda's BettiCurve sampling logic for a single H1 diagram.

    giotto-tda samples evenly between the minimum birth and maximum death values of
    the subdiagram corresponding to the chosen homology dimension.
    """
    h1 = np.asarray(h1, dtype=float)
    if h1.size == 0:
        return np.linspace(0.0, 1.0, n_bins)
    lower = float(np.min(h1))
    upper = float(np.max(h1))
    return np.linspace(lower, upper, n_bins)


def persistence_entropy_manual(
    diagram: np.ndarray,
    normalize: bool = False,
    nan_fill_value: float | None = -1.0,
) -> float:
    """
    Compute persistence entropy from a single diagram using giotto-tda's defaults.

    The calculation uses base-2 Shannon entropy over lifetimes. When the entropy is
    undefined because all lifetimes are zero or the diagram is empty, a configurable
    fill value is returned to mirror giotto-tda's default behavior.
    """
    diagram = np.asarray(diagram, dtype=float)
    if diagram.size == 0:
        value = np.nan
    else:
        lifetimes = diagram[:, 1] - diagram[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            value = float(scipy_entropy(lifetimes, base=2))
            if normalize:
                lifespan_sum = float(np.sum(lifetimes))
                value = value / np.log2(lifespan_sum)
    if nan_fill_value is not None and not np.isfinite(value):
        value = float(nan_fill_value)
    return float(value)


def compute_learner_topology_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute learner-level cycle, transition, and rupture metrics.

    These learner summaries become shared inputs for sections 5.1, 5.2, and 5.3.
    """
    rows = []

    for learner_id, chunk in df.groupby("learner_id", sort=False):
        cloud = get_behavior_point_cloud(df, learner_id)
        group = str(chunk["trajectory_group"].iloc[0])
        final_risk = int(chunk["final_risk"].iloc[0])

        if cloud.shape[0] < 6:
            continue

        rips_diag_full = ripser(cloud, maxdim=1)["dgms"]
        h1_full = finite_h1(rips_diag_full)

        early_cloud = cloud[: max(4, cloud.shape[0] // 2), :]
        late_cloud = cloud[cloud.shape[0] // 2 :, :]

        early_diag = ripser(early_cloud, maxdim=1)["dgms"]
        late_diag = ripser(late_cloud, maxdim=1)["dgms"]
        early_h1 = finite_h1(early_diag)
        late_h1 = finite_h1(late_diag)

        # Stepwise change magnitude along the weekly trajectory.
        steps = np.linalg.norm(np.diff(cloud, axis=0), axis=1)
        mean_step = float(steps.mean()) if len(steps) else 0.0
        max_step = float(steps.max()) if len(steps) else 0.0

        # Transition distance: how strongly the early topology differs from the late topology.
        try:
            transition_distance = float(wasserstein(early_h1, late_h1)) if (len(early_h1) or len(late_h1)) else 0.0
        except Exception:
            transition_distance = 0.0

        # Rupture index: combines sudden local changes, component fragmentation, and
        # a shift toward short-lived late features.
        early_h0 = h0_long_lived_count(early_diag, threshold=0.35)
        late_h0 = h0_long_lived_count(late_diag, threshold=0.35)
        fragmentation_increase = max(0, late_h0 - early_h0)
        late_short_ratio = short_lived_ratio(late_h1)

        rupture_index = (
            0.55 * max_step
            + 0.30 * transition_distance
            + 0.15 * fragmentation_increase
            + 0.20 * late_short_ratio
        )

        rows.append({
            "learner_id": learner_id,
            "trajectory_group": group,
            "final_risk": final_risk,
            "h1_total_persistence": total_persistence(h1_full),
            "h1_max_lifetime": max_lifetime(h1_full),
            "h1_feature_count": int(len(h1_full)),
            "mean_step_change": mean_step,
            "max_step_change": max_step,
            "transition_distance": transition_distance,
            "fragmentation_increase": fragmentation_increase,
            "late_short_lived_ratio": late_short_ratio,
            "rupture_index": rupture_index,
        })

    return pd.DataFrame(rows).sort_values(["trajectory_group", "learner_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Section 5.1: Detection of Cycles in Learning Behavior
# ---------------------------------------------------------------------
def plot_cycle_persistence_by_group(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Plot the distribution of H1 total persistence by trajectory group."""
    order = ["cycle", "transition", "rupture"]
    data = [summary_df.loc[summary_df["trajectory_group"] == g, "h1_total_persistence"].values for g in order]

    plt.figure(figsize=(8, 5))
    bp = plt.boxplot(data, labels=order, patch_artist=True)
    colors = ["#2c7fb8", "#fdae61", "#d7191c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
    plt.ylabel("H1 Total Persistence")
    plt.xlabel("Trajectory Group")
    plt.title("Cycle persistence differs across learner trajectory groups")
    save_png(out_path)


def plot_persistence_diagram_exemplar(point_cloud: np.ndarray, out_path: Path) -> None:
    """Create a Ripser/Persim persistence diagram for the exemplar cyclic learner."""
    result = ripser(point_cloud, maxdim=1)
    diagrams = result["dgms"]
    plt.figure(figsize=(6, 6))
    plot_diagrams(diagrams, show=False)
    plt.title("Persistence diagram for exemplar cyclic learner")
    save_png(out_path)


def plot_topological_barcode_exemplar(point_cloud: np.ndarray, out_path: Path) -> None:
    """Create a GUDHI barcode for the exemplar cyclic learner."""
    _, diag = gudhi_persistence_from_cloud(point_cloud)
    plt.figure(figsize=(8, 4.5))
    gudhi.plot_persistence_barcode(diag)
    plt.title("Topological barcode for exemplar cyclic learner")
    save_png(out_path)


def plot_average_betti_curves(df: pd.DataFrame, out_path: Path) -> None:
    """
    Plot average H1 Betti curves by trajectory group.

    This helps the reader see that cyclic learners preserve loop activity across a wider
    range of filtration values.
    """
    order = ["cycle", "transition", "rupture"]
    all_h1 = []

    learner_h1 = {}
    for learner_id in df["learner_id"].unique():
        cloud = get_behavior_point_cloud(df, learner_id)
        if cloud.shape[0] < 6:
            continue
        h1 = finite_h1(ripser(cloud, maxdim=1)["dgms"])
        learner_h1[learner_id] = h1
        if h1.size > 0:
            all_h1.append(h1[:, 1].max())

    max_death = max(all_h1) if all_h1 else 1.0
    grid = np.linspace(0, max_death, 80)

    plt.figure(figsize=(8, 5))
    for group, color in zip(order, ["#2c7fb8", "#fdae61", "#d7191c"]):
        curves = []
        learners = df.loc[df["trajectory_group"] == group, "learner_id"].unique()
        for learner_id in learners:
            if learner_id not in learner_h1:
                continue
            curves.append(betti_curve_manual(learner_h1[learner_id], grid))
        if curves:
            mean_curve = np.mean(np.vstack(curves), axis=0)
            plt.plot(grid, mean_curve, label=group, linewidth=2.3, color=color)

    plt.xlabel("Filtration value")
    plt.ylabel("Betti-1")
    plt.title("Average Betti curves by trajectory group")
    plt.legend()
    save_png(out_path)


def run_section_5_1(
    processed_df: pd.DataFrame,
    learner_summary_df: pd.DataFrame,
    paths: Dict[str, Path],
) -> None:
    """Generate tables and figures for section 5.1."""
    section_table = (
        learner_summary_df
        .groupby("trajectory_group", as_index=False)[
            ["h1_total_persistence", "h1_max_lifetime", "h1_feature_count"]
        ]
        .agg(["mean", "std"])
    )
    section_table.columns = ["_".join([str(c) for c in col if c]).strip("_") for col in section_table.columns.to_flat_index()]
    save_dataframe(section_table, paths["s51_tab"] / "table_5_1_cycle_summary_by_group.csv")

    # Exemplar cyclic learner = highest H1 total persistence inside the cycle group.
    exemplar_row = (
        learner_summary_df[learner_summary_df["trajectory_group"] == "cycle"]
        .sort_values("h1_total_persistence", ascending=False)
        .iloc[0]
    )
    exemplar_id = str(exemplar_row["learner_id"])
    exemplar_cloud = get_behavior_point_cloud(processed_df, exemplar_id)

    exemplar_table = learner_summary_df.loc[learner_summary_df["learner_id"] == exemplar_id].copy()
    save_dataframe(exemplar_table, paths["s51_tab"] / "table_5_1_exemplar_cyclic_learner.csv")

    plot_cycle_persistence_by_group(
        learner_summary_df,
        paths["s51_fig"] / "figure_5_1_cycle_persistence_by_group.png"
    )
    plot_persistence_diagram_exemplar(
        exemplar_cloud,
        paths["s51_fig"] / "figure_5_1_persistence_diagram_exemplar_cycle.png"
    )
    plot_topological_barcode_exemplar(
        exemplar_cloud,
        paths["s51_fig"] / "figure_5_1_topological_barcode_exemplar_cycle.png"
    )
    plot_average_betti_curves(
        processed_df,
        paths["s51_fig"] / "figure_5_1_betti_curve_by_group.png"
    )


# ---------------------------------------------------------------------
# Section 5.2: Topological Regularization in Deep Learning Models
# ---------------------------------------------------------------------
class AutoEncoder(nn.Module):
    """
    Small autoencoder for weekly learner states.

    The architecture is intentionally lightweight: chapter-quality evidence is the goal,
    not heavy benchmarking.
    """
    def __init__(self, input_dim: int, latent_dim: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def topology_preserving_loss(z: torch.Tensor, x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Lightweight topological regularization surrogate.

    Exact persistent-homology backpropagation would be computationally expensive for a
    chapter demonstration. To keep runtime low while still preserving topology-related
    structure, this surrogate preserves local neighborhood distances from input space
    to latent space.

    In educational terms, if nearby learner states in the original trajectory space are
    pulled apart too strongly in the latent space, the model loses structural continuity.
    This penalty reduces that risk.
    """
    dist_x = torch.cdist(x, x, p=2)
    knn_idx = torch.argsort(dist_x, dim=1)[:, 1 : k + 1]

    target = torch.gather(dist_x, 1, knn_idx)
    target = target / (target.mean() + 1e-8)

    dist_z = torch.cdist(z, z, p=2)
    pred = torch.gather(dist_z, 1, knn_idx)
    pred = pred / (pred.mean() + 1e-8)

    return torch.mean((pred - target) ** 2)


def prepare_model_matrix(processed_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare matrix data for the deep-learning stage.

    The feature set blends behavioral state variables with concept mastery so that the
    latent space can represent both process and content.
    """
    concept_cols = [c for c in processed_df.columns if c.startswith("mastery_")]
    model_df = processed_df[processed_df["keep_for_topology"] == 1].copy()

    feature_cols = [
        "engagement_scaled_smooth",
        "performance_scaled_smooth",
        "help_seeking_scaled_smooth",
        "review_ratio_scaled_smooth",
        "regularity_scaled_smooth",
        "topic_diversity_scaled_smooth",
        "study_ratio_scaled_smooth",
        "practice_ratio_scaled_smooth",
        "quiz_ratio_scaled_smooth",
        "forum_ratio_scaled_smooth",
        "assessment_event",
        "shock_event",
    ] + concept_cols

    X = model_df[feature_cols].to_numpy(dtype=float)
    y = model_df["trajectory_group"].map({"cycle": 0, "transition": 1, "rupture": 2}).to_numpy(dtype=int)
    return X, y, model_df


def compute_latent_audit_metrics(
    latent_cloud: np.ndarray,
    latent_before: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    Compute persistent-homology metrics for a latent point cloud.

    Ripser is used both for fast scalar summaries and for the manual replacements of
    the original giotto-tda Betti-curve and persistence-entropy computations.
    """
    ripser_result = ripser(latent_cloud, maxdim=1)["dgms"]
    h1 = finite_h1(ripser_result)

    metrics = {
        "h1_total_persistence": total_persistence(h1),
        "h1_max_lifetime": max_lifetime(h1),
        "h1_feature_count": float(len(h1)),
    }

    try:
        metrics["persistent_entropy_h1"] = persistence_entropy_manual(
            h1,
            normalize=False,
            nan_fill_value=-1.0,
        )
    except Exception:
        metrics["persistent_entropy_h1"] = np.nan

    if latent_before is not None:
        try:
            before_h1 = finite_h1(ripser(latent_before, maxdim=1)["dgms"])
            metrics["wasserstein_vs_before"] = float(wasserstein(before_h1, h1)) if (len(before_h1) or len(h1)) else 0.0
            metrics["bottleneck_vs_before"] = float(bottleneck(before_h1, h1)) if (len(before_h1) or len(h1)) else 0.0
        except Exception:
            metrics["wasserstein_vs_before"] = np.nan
            metrics["bottleneck_vs_before"] = np.nan

    return metrics


def train_autoencoder(
    X: np.ndarray,
    y: np.ndarray,
    config: PipelineConfig,
    topo_weight: float = 0.0,
    model_name: str = "baseline",
) -> Tuple[AutoEncoder, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train either a baseline or topology-aware autoencoder.

    Returns
    -------
    model
        Trained model.
    audit_df
        Epoch-wise latent topology metrics.
    z_before
        Latent coordinates before training.
    z_after
        Latent coordinates after training.
    """
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_std, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=config.autoencoder_batch_size, shuffle=True)

    model = AutoEncoder(input_dim=X_std.shape[1], latent_dim=config.latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=config.autoencoder_lr)
    mse = nn.MSELoss()

    # Balanced audit sample keeps persistent homology fast while preserving structure.
    audit_idx = []
    rng = np.random.default_rng(config.seed + (13 if model_name == "baseline" else 17))
    for label in np.unique(y):
        label_idx = np.where(y == label)[0]
        take = min(40, len(label_idx))
        audit_idx.extend(rng.choice(label_idx, size=take, replace=False).tolist())
    audit_idx = np.array(sorted(set(audit_idx)), dtype=int)

    def encode_current() -> np.ndarray:
        with torch.no_grad():
            return model.encoder(X_tensor[audit_idx]).cpu().numpy()

    z_before = encode_current()
    audit_labels = y[audit_idx]
    audit_records = []

    before_metrics = compute_latent_audit_metrics(z_before)
    before_metrics.update({"model": model_name, "epoch": 0})
    audit_records.append(before_metrics)

    for epoch in range(1, config.autoencoder_epochs + 1):
        model.train()
        for (batch_x,) in loader:
            optimizer.zero_grad()
            x_hat, z = model(batch_x)
            recon_loss = mse(x_hat, batch_x)
            topo_loss = topology_preserving_loss(z, batch_x, k=5) if topo_weight > 0 else torch.tensor(0.0)
            loss = recon_loss + topo_weight * topo_loss
            loss.backward()
            optimizer.step()

        if epoch % config.audit_every == 0 or epoch == config.autoencoder_epochs:
            z_epoch = encode_current()
            metrics = compute_latent_audit_metrics(z_epoch, latent_before=z_before)
            metrics.update({"model": model_name, "epoch": epoch})
            audit_records.append(metrics)

    model.eval()
    z_after = encode_current()

    audit_df = pd.DataFrame(audit_records)
    return model, audit_df, z_before, z_after, audit_labels


def reconstruction_mse(model: AutoEncoder, X: np.ndarray) -> float:
    """Compute reconstruction MSE on the full standardized dataset."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_std, dtype=torch.float32)
    with torch.no_grad():
        x_hat, _ = model(X_tensor)
    return float(torch.mean((x_hat - X_tensor) ** 2).item())


def latent_silhouette(z: np.ndarray, y: np.ndarray) -> float:
    """Silhouette score in the latent space."""
    if len(np.unique(y)) < 2:
        return np.nan
    return float(silhouette_score(z, y))


def plot_latent_topology_evolution(audit_df: pd.DataFrame, out_path: Path) -> None:
    """Plot epoch-wise H1 total persistence for baseline and topology-aware models."""
    plt.figure(figsize=(8, 5))
    for model_name, color in zip(["baseline", "topology_aware"], ["#7f8c8d", "#2c7fb8"]):
        chunk = audit_df[audit_df["model"] == model_name].sort_values("epoch")
        plt.plot(
            chunk["epoch"],
            chunk["h1_total_persistence"],
            marker="o",
            linewidth=2.2,
            label=model_name.replace("_", " ").title(),
            color=color,
        )
    plt.xlabel("Epoch")
    plt.ylabel("H1 Total Persistence in Latent Space")
    plt.title("Topological evolution of the latent space during training")
    plt.legend()
    save_png(out_path)


def plot_latent_before_after(
    z_before: np.ndarray,
    z_after: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
) -> None:
    """
    Compare latent geometry before and after training.

    A 2-panel figure is appropriate here because the explicit purpose is direct
    comparison before vs. after training.
    """
    colors = {0: "#2c7fb8", 1: "#fdae61", 2: "#d7191c"}
    names = {0: "cycle", 1: "transition", 2: "rupture"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=False, sharey=False)

    for ax, z, title in zip(
        axes,
        [z_before, z_after],
        ["Before training", "After topology-aware training"]
    ):
        for label in sorted(np.unique(labels)):
            idx = np.where(labels == label)[0]
            ax.scatter(
                z[idx, 0],
                z[idx, 1],
                s=26,
                alpha=0.75,
                color=colors[label],
                label=names[label],
            )
        ax.set_title(title)
        ax.set_xlabel("Latent dimension 1")
        ax.set_ylabel("Latent dimension 2")

    handles, labels_text = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="upper center", ncol=3)
    fig.suptitle("Comparison of latent topology before and after training", y=1.02, fontsize=12)
    save_png(out_path)


def plot_betti_before_after(
    z_before: np.ndarray,
    z_after: np.ndarray,
    out_path: Path,
) -> None:
    """
    Plot H1 Betti curves before and after topology-aware training without giotto-tda.

    The curves are computed from Ripser H1 diagrams using the same per-diagram
    sampling idea employed by giotto-tda's BettiCurve transformer.
    """
    h1_before = h1_diagram_for_cloud(z_before)
    h1_after = h1_diagram_for_cloud(z_after)

    grid_before = giotto_like_betti_sampling(h1_before, n_bins=80)
    grid_after = giotto_like_betti_sampling(h1_after, n_bins=80)

    c_before = betti_curve_manual(h1_before, grid_before)
    c_after = betti_curve_manual(h1_after, grid_after)
    x = np.linspace(0, 1, len(c_before))

    plt.figure(figsize=(8, 5))
    plt.plot(x, c_before, linewidth=2.0, label="Before training", color="#7f8c8d")
    plt.plot(x, c_after, linewidth=2.4, label="After topology-aware training", color="#2c7fb8")
    plt.xlabel("Normalized filtration range")
    plt.ylabel("Betti-1")
    plt.title("Before/after Betti curves of the latent space")
    plt.legend()
    save_png(out_path)


def run_section_5_2(
    processed_df: pd.DataFrame,
    paths: Dict[str, Path],
    config: PipelineConfig,
) -> pd.DataFrame:
    """Generate section 5.2 outputs and return a model-ready summary table."""
    X, y, model_df = prepare_model_matrix(processed_df)

    baseline_model, baseline_audit, baseline_before, baseline_after, baseline_labels = train_autoencoder(
        X, y, config, topo_weight=0.0, model_name="baseline"
    )
    topo_model, topo_audit, topo_before, topo_after, topo_labels = train_autoencoder(
        X, y, config, topo_weight=config.topo_weight, model_name="topology_aware"
    )

    audit_df = pd.concat([baseline_audit, topo_audit], axis=0, ignore_index=True)
    save_dataframe(audit_df, paths["s52_tab"] / "table_5_2_latent_topology_evolution.csv")

    # Model comparison table.
    comparison_rows = []
    for model_name, model, z_before, z_after in [
        ("baseline", baseline_model, baseline_before, baseline_after),
        ("topology_aware", topo_model, topo_before, topo_after),
    ]:
        metrics_before = compute_latent_audit_metrics(z_before)
        metrics_after = compute_latent_audit_metrics(z_after, latent_before=z_before)
        comparison_rows.append({
            "model": model_name,
            "reconstruction_mse": reconstruction_mse(model, X),
            "latent_silhouette_after": latent_silhouette(
                z_after,
                baseline_labels if model_name == "baseline" else topo_labels
            ),
            "h1_total_persistence_before": metrics_before["h1_total_persistence"],
            "h1_total_persistence_after": metrics_after["h1_total_persistence"],
            "h1_max_lifetime_after": metrics_after["h1_max_lifetime"],
            "persistent_entropy_h1_after": metrics_after["persistent_entropy_h1"],
            "wasserstein_vs_before": metrics_after.get("wasserstein_vs_before", np.nan),
            "bottleneck_vs_before": metrics_after.get("bottleneck_vs_before", np.nan),
        })
    comparison_df = pd.DataFrame(comparison_rows)
    save_dataframe(comparison_df, paths["s52_tab"] / "table_5_2_model_comparison.csv")

    plot_latent_topology_evolution(
        audit_df,
        paths["s52_fig"] / "figure_5_2_latent_topology_evolution.png"
    )
    plot_latent_before_after(
        topo_before,
        topo_after,
        topo_labels,
        paths["s52_fig"] / "figure_5_2_latent_space_before_after_training.png"
    )
    plot_betti_before_after(
        topo_before,
        topo_after,
        paths["s52_fig"] / "figure_5_2_topology_before_after_training_betti.png"
    )

    return comparison_df


# ---------------------------------------------------------------------
# Section 5.3: Topology-Based Educational Recommendation System
# ---------------------------------------------------------------------
def build_final_state_dataframe(processed_df: pd.DataFrame, learner_summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract final-week learner states and attach topology-based risk signals.

    This table is the decision-time representation for the recommendation system.
    """
    final_df = processed_df.sort_values("week").groupby("learner_id", as_index=False).tail(1).copy()
    final_df = final_df.reset_index(drop=True)
    final_df["point_index"] = np.arange(len(final_df), dtype=int)

    final_df = final_df.merge(
        learner_summary_df[[
            "learner_id", "h1_total_persistence", "transition_distance",
            "rupture_index", "late_short_lived_ratio", "fragmentation_increase"
        ]],
        on="learner_id",
        how="left",
    )

    concept_cols = [c for c in final_df.columns if c.startswith("mastery_")]
    final_df["mean_mastery"] = final_df[concept_cols].mean(axis=1)

    # Topology-aware assistive risk score:
    # high rupture and fragmentation increase risk, while stable cyclicity
    # and stronger mastery reduce it.
    raw_score = (
        1.40 * final_df["rupture_index"].fillna(0.0)
        + 0.55 * final_df["fragmentation_increase"].fillna(0.0)
        + 0.85 * (1.0 - final_df["mean_mastery"])
        + 0.35 * final_df["late_short_lived_ratio"].fillna(0.0)
        - 0.30 * final_df["h1_total_persistence"].fillna(0.0)
    )
    final_df["assistive_risk_score"] = raw_score
    final_df["assistive_risk_probability"] = sigmoid((raw_score - raw_score.mean()) / (raw_score.std() + 1e-8))
    return final_df


def build_mapper_graph(
    final_df: pd.DataFrame,
    concept_cols: Sequence[str],
    paths: Dict[str, Path],
    config: PipelineConfig
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Build a Mapper graph over final learner states.

    The graph serves as a topological knowledge map: nodes gather learners with similar
    mastery/behavior profiles, while edges encode overlap between neighboring local clusters.
    """
    feature_cols = [
        "engagement_scaled_smooth", "performance_scaled_smooth", "help_seeking_scaled_smooth",
        "review_ratio_scaled_smooth", "regularity_scaled_smooth", "topic_diversity_scaled_smooth",
    ] + list(concept_cols)

    X = final_df[feature_cols].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X)

    # Use a 2D PCA lens for speed and stability.
    lens = PCA(n_components=2, random_state=config.seed).fit_transform(X)
    lens = MinMaxScaler().fit_transform(lens)

    mapper = km.KeplerMapper(verbose=0)
    graph = mapper.map(
        lens,
        X=X,
        cover=km.Cover(n_cubes=config.mapper_n_cubes, perc_overlap=config.mapper_overlap),
        clusterer=DBSCAN(eps=0.95, min_samples=3),
    )

    html_path = paths["s53_fig"] / "figure_5_3_mapper_knowledge_map.html"
    try:
        mapper.visualize(
            graph,
            path_html=str(html_path),
            title="Topological Knowledge Map of Final Learner States"
        )
    except Exception:
        # HTML export is helpful but not essential for the chapter figure.
        pass

    return graph, X, lens


def mapper_graph_to_networkx(graph: dict, risk_values: np.ndarray) -> nx.Graph:
    """
    Convert a KeplerMapper graph dictionary into a NetworkX graph for static figure export.
    """
    G = nx.Graph()

    node_members = graph.get("nodes", {})
    links = graph.get("links", {})

    for node_id, members in node_members.items():
        members = list(members)
        mean_risk = float(np.mean(risk_values[members])) if members else 0.0
        G.add_node(node_id, size=len(members), mean_risk=mean_risk, members=members)

    if isinstance(links, dict):
        for src, dst_values in links.items():
            if isinstance(dst_values, dict):
                targets = list(dst_values.keys())
            elif isinstance(dst_values, (list, set, tuple)):
                targets = list(dst_values)
            else:
                targets = [dst_values]
            for dst in targets:
                if src in G.nodes and dst in G.nodes:
                    G.add_edge(src, dst)

    return G


def plot_mapper_knowledge_map(
    graph: dict,
    final_df: pd.DataFrame,
    out_path: Path,
) -> Tuple[nx.Graph, Dict[int, List[str]]]:
    """
    Plot the Mapper graph as a static PNG suitable for chapter insertion.
    """
    risk_values = final_df["assistive_risk_probability"].to_numpy(dtype=float)
    G = mapper_graph_to_networkx(graph, risk_values)

    if G.number_of_nodes() == 0:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "Mapper graph did not produce nodes under current settings.", ha="center", va="center")
        plt.axis("off")
        save_png(out_path)
        return G, {}

    pos = nx.spring_layout(G, seed=42, k=0.65)
    node_sizes = [300 + 55 * G.nodes[n]["size"] for n in G.nodes()]
    node_colors = [G.nodes[n]["mean_risk"] for n in G.nodes()]

    plt.figure(figsize=(9, 6))
    nx.draw_networkx_edges(G, pos, alpha=0.35, width=1.1)
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="viridis",
        alpha=0.92,
    )
    plt.colorbar(nodes, label="Mean assistive risk probability")
    plt.title("Topological knowledge map of learner states (Mapper)")
    plt.axis("off")
    save_png(out_path)

    # Reverse lookup: point index -> list of mapper nodes containing it.
    point_membership: Dict[int, List[str]] = defaultdict(list)
    for node_id, members in graph.get("nodes", {}).items():
        for idx in members:
            point_membership[int(idx)].append(node_id)

    return G, point_membership


def extract_successful_triads(final_df: pd.DataFrame, concept_cols: Sequence[str], top_k: int = 12) -> List[Tuple[str, str, str]]:
    """
    Build maximal concept simplices from successful learners.

    A simplex here represents a set of concepts frequently co-mastered in strong final
    states. This supports the educational reading of topological knowledge structures.
    """
    concept_names = [c.replace("mastery_", "") for c in concept_cols]
    triad_counter = Counter()

    successful = final_df[
        (final_df["assistive_risk_probability"] < final_df["assistive_risk_probability"].median())
        & (final_df["mean_mastery"] >= final_df["mean_mastery"].median())
    ].copy()

    for _, row in successful.iterrows():
        mastered = []
        for col, concept in zip(concept_cols, concept_names):
            if float(row[col]) >= 0.72:
                mastered.append(concept)
        if len(mastered) >= 3:
            for triad in combinations(sorted(mastered), 3):
                triad_counter[triad] += 1

    return [triad for triad, _ in triad_counter.most_common(top_k)]


def build_toponetx_learning_graph(
    final_df: pd.DataFrame,
    concept_cols: Sequence[str],
) -> Tuple[object, nx.Graph, pd.DataFrame]:
    """
    Build a TopoNetX simplicial complex and derive its concept-level learning graph.
    """
    concepts, prereq = get_concepts_and_prerequisites()
    maximal_simplices = list(extract_successful_triads(final_df, concept_cols, top_k=12))

    # Ensure prerequisite pathways also appear.
    for child, parents in prereq.items():
        if parents:
            if len(parents) == 1:
                maximal_simplices.append(tuple(sorted([parents[0], child, "Event Logs"] if parents[0] != "Event Logs" else [parents[0], child, "Temporal Windows"])))
            elif len(parents) >= 2:
                maximal_simplices.append(tuple(sorted([parents[0], parents[1], child])))

    maximal_simplices = [tuple(simplex) for simplex in maximal_simplices if len(set(simplex)) >= 3]
    maximal_simplices = list(dict.fromkeys([tuple(sorted(s)) for s in maximal_simplices]))

    SC = tnx.SimplicialComplex(maximal_simplices)

    # Incidence summary for table export.
    incidence_shape = None
    try:
        rows_idx, cols_idx, B01 = SC.incidence_matrix(rank=1, index=True)
        incidence_shape = B01.shape
    except Exception:
        try:
            rows_idx, cols_idx, B01 = SC.incidence_matrix(1, index=True)
            incidence_shape = B01.shape
        except Exception:
            incidence_shape = (np.nan, np.nan)

    G = nx.Graph()
    for concept in concepts:
        G.add_node(concept)

    edge_counter = Counter()
    for simplex in maximal_simplices:
        for u, v in combinations(simplex, 2):
            edge_counter[tuple(sorted((u, v)))] += 1

    for (u, v), weight in edge_counter.items():
        G.add_edge(u, v, weight=weight)

    centrality = nx.degree_centrality(G)
    graph_table = pd.DataFrame({
        "concept": list(G.nodes()),
        "degree": [G.degree(n) for n in G.nodes()],
        "degree_centrality": [centrality[n] for n in G.nodes()],
        "in_incidence_structure": [1 if G.degree(n) > 0 else 0 for n in G.nodes()],
        "incidence_rows": incidence_shape[0],
        "incidence_cols": incidence_shape[1],
    }).sort_values(["degree", "degree_centrality"], ascending=False)

    return SC, G, graph_table


def plot_toponetx_learning_graph(G: nx.Graph, out_path: Path) -> None:
    """
    Plot the concept-level learning graph derived from TopoNetX simplices.
    """
    plt.figure(figsize=(9, 6))
    pos = nx.spring_layout(G, seed=42, k=0.75)
    weights = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float) if G.number_of_edges() else np.array([])
    widths = 1.0 + (weights / weights.max() * 3.0 if len(weights) else weights)

    degree_centrality = nx.degree_centrality(G)
    node_colors = np.array([degree_centrality[n] for n in G.nodes()], dtype=float)
    node_sizes = np.array([700 + 1200 * degree_centrality[n] for n in G.nodes()], dtype=float)

    nx.draw_networkx_edges(G, pos, width=widths if len(widths) else 1.0, alpha=0.35, edge_color="#7f8c8d")
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap="plasma", alpha=0.95)
    nx.draw_networkx_labels(G, pos, font_size=9)

    plt.colorbar(nodes, label="Concept centrality")
    plt.title("Topological learning graph derived from concept simplices")
    plt.axis("off")
    save_png(out_path)


def concept_prereq_satisfaction(row: pd.Series, concept: str, prereq: Dict[str, List[str]]) -> float:
    """Average mastery of prerequisites for a candidate concept."""
    parents = prereq.get(concept, [])
    if not parents:
        return 1.0
    values = [float(row[f"mastery_{p}"]) for p in parents]
    return float(np.mean(values))


def generate_topological_recommendations(
    final_df: pd.DataFrame,
    mapper_graph: dict,
    point_membership: Dict[int, List[str]],
    concept_graph: nx.Graph,
) -> pd.DataFrame:
    """
    Generate actionable recommendations by combining:
    - a learner's current mastery gaps,
    - nearby Mapper neighborhoods with stronger mastery,
    - concept centrality in the learning graph,
    - prerequisite satisfaction.

    The recommendations are assistive and interpretable, matching the chapter's stance.
    """
    concepts, prereq = get_concepts_and_prerequisites()
    concept_cols = [f"mastery_{c}" for c in concepts]
    graph_nodes = mapper_graph.get("nodes", {})

    # Precompute node-level concept profiles.
    node_profiles = {}
    node_risk = {}
    for node_id, members in graph_nodes.items():
        sub = final_df.iloc[list(members)]
        node_profiles[node_id] = sub[concept_cols].mean(axis=0).to_dict()
        node_risk[node_id] = float(sub["assistive_risk_probability"].mean())

    centrality = nx.degree_centrality(concept_graph)

    recommendations = []
    ranked = final_df.sort_values("assistive_risk_probability", ascending=False).reset_index(drop=True)

    for row_idx, row in ranked.head(12).iterrows():
        member_nodes = point_membership.get(int(row["point_index"]), [])

        # Candidate reference nodes = current nodes + their graph neighbors,
        # but only if the neighboring node shows lower mean risk.
        reference_nodes = set()
        for node_id in member_nodes:
            reference_nodes.add(node_id)
            links = mapper_graph.get("links", {}).get(node_id, [])
            if isinstance(links, dict):
                links = list(links.keys())
            for neigh in links:
                if node_risk.get(neigh, 1.0) < node_risk.get(node_id, 1.0):
                    reference_nodes.add(neigh)

        if not reference_nodes:
            # Fall back to globally stronger states.
            low_risk = final_df[final_df["assistive_risk_probability"] <= final_df["assistive_risk_probability"].median()]
            reference_profile = low_risk[concept_cols].mean(axis=0).to_dict()
        else:
            node_profile_df = pd.DataFrame([node_profiles[n] for n in reference_nodes if n in node_profiles])
            reference_profile = node_profile_df.mean(axis=0).to_dict()

        candidate_scores = []
        for concept in concepts:
            mastery_now = float(row[f"mastery_{concept}"])
            gap = 1.0 - mastery_now
            prereq_score = concept_prereq_satisfaction(row, concept, prereq)
            topo_support = float(reference_profile.get(f"mastery_{concept}", reference_profile.get(concept, 0.0)))
            graph_support = float(centrality.get(concept, 0.0))

            # The score rewards unmet need, local topological support, and prerequisite readiness.
            score = 0.45 * gap + 0.25 * topo_support + 0.20 * prereq_score + 0.10 * graph_support

            if mastery_now < 0.80:
                candidate_scores.append((concept, score, mastery_now, prereq_score, topo_support))

        candidate_scores = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        top3 = candidate_scores[:3]

        recommendations.append({
            "learner_id": row["learner_id"],
            "trajectory_group": row["trajectory_group"],
            "assistive_risk_probability": float(row["assistive_risk_probability"]),
            "current_mapper_nodes": "; ".join(member_nodes) if member_nodes else "global-reference",
            "recommended_concepts": "; ".join([x[0] for x in top3]),
            "current_mastery": "; ".join([f"{x[0]}={x[2]:.2f}" for x in top3]),
            "prerequisite_support": "; ".join([f"{x[0]}={x[3]:.2f}" for x in top3]),
            "topological_support": "; ".join([f"{x[0]}={x[4]:.2f}" for x in top3]),
            "rationale": (
                "Recommended from nearby low-risk topological neighborhoods and the "
                "concept learning graph, prioritizing unmet concepts whose prerequisites "
                "are sufficiently supported."
            ),
        })

    return pd.DataFrame(recommendations)


def run_section_5_3(
    processed_df: pd.DataFrame,
    learner_summary_df: pd.DataFrame,
    paths: Dict[str, Path],
    config: PipelineConfig,
) -> None:
    """Generate section 5.3 outputs."""
    final_df = build_final_state_dataframe(processed_df, learner_summary_df)
    concept_cols = [c for c in final_df.columns if c.startswith("mastery_")]

    mapper_graph, X_final, lens = build_mapper_graph(final_df, concept_cols, paths, config)
    static_G, point_membership = plot_mapper_knowledge_map(
        mapper_graph,
        final_df,
        paths["s53_fig"] / "figure_5_3_mapper_knowledge_map.png"
    )

    SC, concept_graph, graph_table = build_toponetx_learning_graph(final_df, concept_cols)
    save_dataframe(graph_table, paths["s53_tab"] / "table_5_3_concept_graph_summary.csv")

    plot_toponetx_learning_graph(
        concept_graph,
        paths["s53_fig"] / "figure_5_3_topological_learning_graph.png"
    )

    recommendation_df = generate_topological_recommendations(
        final_df,
        mapper_graph,
        point_membership,
        concept_graph,
    )
    save_dataframe(recommendation_df, paths["s53_tab"] / "table_5_3_actionable_recommendations.csv")


# ---------------------------------------------------------------------
# Shared reporting helpers
# ---------------------------------------------------------------------
def save_shared_data(
    hidden_df: pd.DataFrame,
    events_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    learner_summary_df: pd.DataFrame,
    paths: Dict[str, Path]
) -> None:
    """Save the intermediate data products used by the full chapter pipeline."""
    save_dataframe(hidden_df, paths["shared_data"] / "synthetic_hidden_weekly_profiles.csv")
    save_dataframe(events_df, paths["shared_data"] / "synthetic_event_logs.csv")
    save_dataframe(weekly_df, paths["shared_data"] / "weekly_states_from_event_logs.csv")
    save_dataframe(processed_df, paths["shared_intermediate"] / "weekly_states_preprocessed.csv")
    save_dataframe(learner_summary_df, paths["shared_intermediate"] / "learner_topology_summary.csv")


def write_readme(paths: Dict[str, Path], config: PipelineConfig) -> None:
    """Write a concise README to document the output structure."""
    readme = f"""
Topological chapter pipeline outputs
====================================

This folder was created by chapter_topology_pipeline_complete.py

Section folders
---------------
5_1_detection_of_cycles
    figures/
        figure_5_1_cycle_persistence_by_group.png
        figure_5_1_persistence_diagram_exemplar_cycle.png
        figure_5_1_topological_barcode_exemplar_cycle.png
        figure_5_1_betti_curve_by_group.png
    tables/
        table_5_1_cycle_summary_by_group.csv
        table_5_1_exemplar_cyclic_learner.csv

5_2_topological_regularization
    figures/
        figure_5_2_latent_topology_evolution.png
        figure_5_2_latent_space_before_after_training.png
        figure_5_2_topology_before_after_training_betti.png
    tables/
        table_5_2_latent_topology_evolution.csv
        table_5_2_model_comparison.csv

5_3_topological_recommendations
    figures/
        figure_5_3_mapper_knowledge_map.png
        figure_5_3_mapper_knowledge_map.html
        figure_5_3_topological_learning_graph.png
    tables/
        table_5_3_concept_graph_summary.csv
        table_5_3_actionable_recommendations.csv

shared/
    data/
        synthetic_hidden_weekly_profiles.csv
        synthetic_event_logs.csv
        weekly_states_from_event_logs.csv
    intermediate/
        weekly_states_preprocessed.csv
        learner_topology_summary.csv

Core configuration
------------------
seed={config.seed}
n_learners_per_group={config.n_learners_per_group}
n_weeks={config.n_weeks}
n_concepts={config.n_concepts}
autoencoder_epochs={config.autoencoder_epochs}
topo_weight={config.topo_weight}

Notes
-----
- The script is designed to be reasonably fast.
- Persistent homology is computed only in H0 and H1.
- The deep model uses a lightweight topology-preserving regularization surrogate
  during optimization and uses persistent homology for auditing the latent space.
"""
    (paths["root"] / "README_outputs.txt").write_text(readme.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------
def parse_args() -> PipelineConfig:
    """Parse command-line arguments into a PipelineConfig."""
    parser = argparse.ArgumentParser(description="Complete topological pipeline for chapter sections 5.1, 5.2, and 5.3.")
    parser.add_argument("--output-root", type=str, default="chapter_outputs", help="Root folder for all outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-learners-per-group", type=int, default=24, help="Learners per synthetic archetype.")
    parser.add_argument("--n-weeks", type=int, default=12, help="Number of course weeks.")
    parser.add_argument("--autoencoder-epochs", type=int, default=18, help="Training epochs for the deep model.")
    parser.add_argument("--topo-weight", type=float, default=0.30, help="Weight for topology-preserving regularization.")
    args = parser.parse_args()

    return PipelineConfig(
        seed=args.seed,
        n_learners_per_group=args.n_learners_per_group,
        n_weeks=args.n_weeks,
        autoencoder_epochs=args.autoencoder_epochs,
        topo_weight=args.topo_weight,
        output_root=args.output_root,
    )


def main() -> None:
    """Run the full chapter pipeline."""
    config = parse_args()
    check_dependencies()
    set_global_seed(config.seed)

    paths = ensure_output_tree(config.output_root)

    # 1) Synthetic data generation
    hidden_df = simulate_hidden_weekly_profiles(config)
    events_df = generate_event_logs(hidden_df, config)
    weekly_df = aggregate_weekly_states(events_df, hidden_df)
    processed_df = preprocess_weekly_states(weekly_df, config)

    # 2) Learner-level shared topology summary
    learner_summary_df = compute_learner_topology_summary(processed_df)

    # 3) Save shared data products
    save_shared_data(hidden_df, events_df, weekly_df, processed_df, learner_summary_df, paths)

    # 4) Section-specific outputs
    run_section_5_1(processed_df, learner_summary_df, paths)
    run_section_5_2(processed_df, paths, config)
    run_section_5_3(processed_df, learner_summary_df, paths, config)

    # 5) Output README
    write_readme(paths, config)

    print(f"Pipeline completed successfully. Outputs written to: {Path(config.output_root).resolve()}")


if __name__ == "__main__":
    main()
