#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Topological regularization for educational recommendation and personalized tutoring
===================================================================================

This script is a self-contained, reproducible experimental pipeline aligned with
Sections 1, 2, 3, and 4 of the uploaded chapter draft
"Topological Regularization in Educational Recommendation and Personalized Tutoring".

The code operationalizes the four central design components described in the draft:

1. Structural modeling
   - learner-resource interaction logs
   - concept prerequisite graph
   - resource transition / co-usage graph
   - higher-order simplicial structure over concept sequences

2. Topology-aware learning
   - a baseline recommendation model
   - a topology-regularized recommendation model implemented in PyTorch
   - graph smoothness regularization over learners, resources, and concepts
   - prerequisite-consistency and neighborhood-stability penalties

3. Topology-constrained decision rules
   - plain top-k recommendation
   - topology-constrained reranking for prerequisite consistency and bounded step size
   - short tutoring rollouts for sequential evaluation

4. Topology-based auditing
   - persistent homology with Ripser
   - persistence visualizations with Persim
   - Mapper graphs with KeplerMapper
   - higher-order structure summaries with TopoNetX
   - subgroup and robustness audits designed for manuscript-ready reporting

Important implementation choice
-------------------------------
No external educational dataset was attached with the chapter draft. Therefore,
this script generates a synthetic but pedagogically structured dataset that is
explicitly designed to reflect the chapter's rationale:

- sparse, sequential educational interactions
- prerequisite-governed concept progressions
- heterogeneous learners
- support-oriented resources such as videos and tutoring interventions
- realistic tensions between personalization and curricular coherence

Output contract for the manuscript
----------------------------------
The script creates one root results folder with three subsections matching the
requested manuscript structure:

    5_1_structural_modeling/
    5_2_topology_aware_learning/
    5_3_topology_constrained_decisions/

Each subsection contains figures/ and tables/ folders so that the outputs can be
inserted directly into Sections 5.1, 5.2, and 5.3 of the chapter.
A shared manifest is also produced with suggested roles for each artifact.

Required packages
-----------------
Core scientific stack:
    numpy, pandas, matplotlib, networkx, scipy, scikit-learn, torch

Topological stack used in this script:
    ripser, persim, kmapper, toponetx

Install example (GUDHI is intentionally NOT used):
    pip install numpy pandas matplotlib networkx scipy scikit-learn torch \
                ripser persim kmapper toponetx

Usage
-----
Run with default settings:
    python topological_regularization_educational_recommendation.py

Optional example with custom output folder and seed:
    python topological_regularization_educational_recommendation.py \
        --output_dir results_chapter_5 --random_state 7

Notes for adaptation to real data
---------------------------------
The script is intentionally organized so that a real LMS / tutoring dataset can
be substituted later. The synthetic generator can be replaced by a CSV loader,
while the remaining pipeline (graph construction, model training, auditing, and
report generation) can remain almost unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Dependency checks for the topological libraries explicitly requested by the user.
# -----------------------------------------------------------------------------
MISSING_PACKAGES: List[str] = []

try:
    import kmapper as km
except Exception:  # pragma: no cover - handled explicitly at runtime
    km = None
    MISSING_PACKAGES.append("kmapper")

try:
    import toponetx as tnx
except Exception:  # pragma: no cover - handled explicitly at runtime
    tnx = None
    MISSING_PACKAGES.append("toponetx")

try:
    from ripser import ripser
except Exception:  # pragma: no cover - handled explicitly at runtime
    ripser = None
    MISSING_PACKAGES.append("ripser")

try:
    from persim import PersistenceLandscaper, plot_diagrams, wasserstein
except Exception:  # pragma: no cover - handled explicitly at runtime
    PersistenceLandscaper = None
    plot_diagrams = None
    wasserstein = None
    MISSING_PACKAGES.append("persim")

# -----------------------------------------------------------------------------
# Global constants
# -----------------------------------------------------------------------------
MODALITIES = ["reading", "video", "practice", "quiz", "tutoring"]
SUPPORTIVE_MODALITIES = {"video", "tutoring"}
ASSESSMENT_MODALITIES = {"practice", "quiz"}
PROFILE_ORDER = ["underprepared", "regular", "advanced"]
PROFILE_TO_INDEX = {name: idx for idx, name in enumerate(PROFILE_ORDER)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Central configuration object.

    The values below intentionally balance three goals:
    1. Keep the script self-contained and fast enough to run on CPU.
    2. Preserve enough structural richness to make the topological analysis meaningful.
    3. Generate a compact but publishable set of outputs for manuscript Section 5.
    """

    output_dir: str = "results_topological_education"
    random_state: int = 42

    # Synthetic curriculum and data generation
    n_learners_train: int = 180
    n_learners_rollout: int = 72
    min_interactions: int = 18
    max_interactions: int = 28
    mastery_threshold: float = 0.60

    # Learning model
    embedding_dim: int = 16
    hidden_dim: int = 64
    batch_size: int = 512
    epochs: int = 28
    learning_rate: float = 0.003
    weight_decay: float = 1e-5
    negative_ratio: int = 3

    # Regularization strengths
    lambda_learner_smooth: float = 0.040
    lambda_resource_smooth: float = 0.055
    lambda_concept_smooth: float = 0.070
    lambda_alignment: float = 0.045
    lambda_prereq: float = 0.120
    lambda_stability: float = 0.060

    # Evaluation
    top_k: int = 10
    rollout_steps: int = 12
    ph_sample_size: int = 120
    betti_grid_size: int = 160

    # Mapper settings
    mapper_n_cubes: int = 6
    mapper_perc_overlap: float = 0.35
    mapper_dbscan_min_samples: int = 3


# -----------------------------------------------------------------------------
# General utility functions
# -----------------------------------------------------------------------------
def check_dependencies() -> None:
    """Raise a clear installation message when required packages are missing."""
    if MISSING_PACKAGES:
        missing = ", ".join(sorted(set(MISSING_PACKAGES)))
        raise ImportError(
            "Missing required packages: "
            f"{missing}.\n"
            "Install them, for example, with:\n"
            "pip install numpy pandas matplotlib networkx scipy scikit-learn torch "
            "ripser persim kmapper toponetx\n"
            "This script intentionally does NOT use GUDHI."
        )


def set_global_seed(seed: int) -> None:
    """Set every relevant random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ArtifactLogger:
    """Collect metadata about every figure/table created by the script.

    This helper is designed for the manuscript workflow. The resulting manifest
    tells the author what each file is meant to support in the written analysis.
    """

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(
        self,
        section: str,
        kind: str,
        file_path: Path,
        role: str,
        suggested_caption: str,
        interpretation_hint: str,
    ) -> None:
        self.rows.append(
            {
                "section": section,
                "type": kind,
                "file": str(file_path),
                "role": role,
                "suggested_caption": suggested_caption,
                "interpretation_hint": interpretation_hint,
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def ensure_output_tree(root: str) -> Dict[str, Path]:
    """Create the directory tree aligned with manuscript Sections 5.1–5.3."""
    root_path = Path(root)
    paths = {
        "root": root_path,
        "shared": root_path / "shared",
        "shared_fig": root_path / "shared" / "figures",
        "shared_tab": root_path / "shared" / "tables",
        "s51": root_path / "5_1_structural_modeling",
        "s51_fig": root_path / "5_1_structural_modeling" / "figures",
        "s51_tab": root_path / "5_1_structural_modeling" / "tables",
        "s52": root_path / "5_2_topology_aware_learning",
        "s52_fig": root_path / "5_2_topology_aware_learning" / "figures",
        "s52_tab": root_path / "5_2_topology_aware_learning" / "tables",
        "s53": root_path / "5_3_topology_constrained_decisions",
        "s53_fig": root_path / "5_3_topology_constrained_decisions" / "figures",
        "s53_tab": root_path / "5_3_topology_constrained_decisions" / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_figure(path: Path, dpi: int = 220) -> None:
    """Standardized figure export used everywhere in the script."""
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def clip01(x: np.ndarray | float) -> np.ndarray | float:
    """Clip scalars or arrays to the closed unit interval."""
    return np.clip(x, 0.0, 1.0)


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    """Return the arithmetic mean or a default when the input is empty."""
    return float(np.mean(values)) if len(values) > 0 else float(default)


def profile_one_hot(profile: str) -> np.ndarray:
    """Encode learner profile as a length-3 one-hot vector."""
    out = np.zeros(len(PROFILE_ORDER), dtype=float)
    out[PROFILE_TO_INDEX[profile]] = 1.0
    return out


# -----------------------------------------------------------------------------
# Curriculum, resources, and learners
# -----------------------------------------------------------------------------
def build_concept_graph() -> Tuple[nx.DiGraph, pd.DataFrame, Dict[int, Tuple[float, float]]]:
    """Construct a compact curriculum graph with prerequisite structure.

    The graph is deliberately designed as a pedagogically interpretable DAG:
    foundational concepts branch into algebraic/modeling and data-literacy paths,
    and a capstone concept depends on multiple strands.
    """
    concepts = [
        (0, "Number Sense", "Foundations", 0),
        (1, "Fractions and Ratios", "Foundations", 1),
        (2, "Linear Equations", "Algebra", 2),
        (3, "Systems of Equations", "Algebra", 3),
        (4, "Functions", "Algebra", 4),
        (5, "Quadratic Models", "Modeling", 5),
        (6, "Derivatives", "Modeling", 6),
        (7, "Optimization", "Modeling", 7),
        (8, "Data Literacy", "Data", 2),
        (9, "Probability", "Data", 3),
        (10, "Statistical Inference", "Data", 4),
        (11, "Integrated Modeling Project", "Capstone", 8),
    ]

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (1, 8),
        (8, 9),
        (9, 10),
        (4, 11),
        (5, 11),
        (7, 11),
        (10, 11),
    ]

    graph = nx.DiGraph()
    for concept_id, name, strand, depth in concepts:
        graph.add_node(concept_id, name=name, strand=strand, depth=depth)
    graph.add_edges_from(edges)

    # Manual layout keeps the curriculum visually stable across runs.
    strand_y = {
        "Foundations": 2.0,
        "Algebra": 1.2,
        "Modeling": 0.5,
        "Data": -0.6,
        "Capstone": 1.9,
    }
    pos = {}
    for concept_id, name, strand, depth in concepts:
        y = strand_y[strand]
        if strand == "Data":
            x = depth
        elif strand == "Capstone":
            x = 8.8
        else:
            x = depth
        pos[concept_id] = (x, y)

    concept_df = pd.DataFrame(concepts, columns=["concept_idx", "concept_name", "strand", "depth"])
    return graph, concept_df, pos


def build_resource_catalog(concept_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create a resource catalog with one item per modality for each concept.

    This design gives every concept a support-oriented and an assessment-oriented
    set of resources, which is useful for analyzing prerequisite-safe tutoring.
    """
    modality_offsets = {
        "reading": 0.02,
        "video": -0.04,
        "practice": 0.03,
        "quiz": 0.08,
        "tutoring": -0.06,
    }
    duration_by_modality = {
        "reading": 14,
        "video": 10,
        "practice": 16,
        "quiz": 12,
        "tutoring": 18,
    }

    rows = []
    resource_idx = 0
    for row in concept_df.itertuples(index=False):
        for modality in MODALITIES:
            depth_factor = 0.20 + 0.06 * row.depth
            difficulty = clip01(depth_factor + modality_offsets[modality] + rng.normal(0.0, 0.03))
            supportive_flag = float(modality in SUPPORTIVE_MODALITIES)
            assessment_flag = float(modality in ASSESSMENT_MODALITIES)
            rows.append(
                {
                    "resource_idx": resource_idx,
                    "resource_id": f"R{resource_idx:03d}",
                    "resource_name": f"{row.concept_name} — {modality.title()}",
                    "concept_idx": int(row.concept_idx),
                    "concept_name": row.concept_name,
                    "strand": row.strand,
                    "depth": int(row.depth),
                    "modality": modality,
                    "modality_idx": MODALITIES.index(modality),
                    "difficulty": float(difficulty),
                    "expected_duration": int(duration_by_modality[modality] + rng.integers(-2, 3)),
                    "supportive_flag": supportive_flag,
                    "assessment_flag": assessment_flag,
                }
            )
            resource_idx += 1
    return pd.DataFrame(rows)


def sample_modality_preferences(profile: str, rng: np.random.Generator) -> np.ndarray:
    """Generate learner modality preferences conditioned on preparation profile."""
    if profile == "underprepared":
        alpha = np.array([1.8, 2.6, 1.6, 0.8, 2.8])
    elif profile == "advanced":
        alpha = np.array([1.0, 1.2, 2.8, 2.4, 0.9])
    else:
        alpha = np.array([1.6, 1.8, 2.2, 1.7, 1.4])
    prefs = rng.dirichlet(alpha)
    return prefs.astype(float)


@dataclass
class LearnerSpec:
    """Static learner attributes used to initialize synthetic educational states."""

    learner_idx: int
    learner_id: str
    profile: str
    support_need: float
    perseverance: float
    modality_preferences: np.ndarray
    initial_mastery: np.ndarray


def generate_learner_specs(n_learners: int, n_concepts: int, rng: np.random.Generator) -> List[LearnerSpec]:
    """Create synthetic learners with heterogeneous background preparation.

    The initialization intentionally makes the logs sparse and heterogeneous,
    which is precisely the setting where structural regularization is useful.
    """
    specs: List[LearnerSpec] = []
    profile_probs = [0.36, 0.44, 0.20]
    for learner_idx in range(n_learners):
        profile = rng.choice(PROFILE_ORDER, p=profile_probs)
        if profile == "underprepared":
            support_need = clip01(rng.normal(0.78, 0.10))
            perseverance = clip01(rng.normal(0.62, 0.10))
            base = np.linspace(0.55, 0.12, n_concepts)
        elif profile == "advanced":
            support_need = clip01(rng.normal(0.18, 0.08))
            perseverance = clip01(rng.normal(0.80, 0.08))
            base = np.linspace(0.78, 0.28, n_concepts)
        else:
            support_need = clip01(rng.normal(0.40, 0.10))
            perseverance = clip01(rng.normal(0.72, 0.10))
            base = np.linspace(0.66, 0.20, n_concepts)

        initial_mastery = clip01(base + rng.normal(0.0, 0.06, size=n_concepts))
        modality_preferences = sample_modality_preferences(profile, rng)
        specs.append(
            LearnerSpec(
                learner_idx=learner_idx,
                learner_id=f"L{learner_idx:03d}",
                profile=profile,
                support_need=float(support_need),
                perseverance=float(perseverance),
                modality_preferences=modality_preferences,
                initial_mastery=initial_mastery.astype(float),
            )
        )
    return specs


# -----------------------------------------------------------------------------
# Educational state dynamics
# -----------------------------------------------------------------------------
def build_prerequisite_maps(graph: nx.DiGraph) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Return direct prerequisite lists and all-ancestor prerequisite lists."""
    direct = {node: sorted(list(graph.predecessors(node))) for node in graph.nodes()}
    ancestors = {node: sorted(list(nx.ancestors(graph, node))) for node in graph.nodes()}
    return direct, ancestors


def compute_concept_distance_matrix(graph: nx.DiGraph) -> np.ndarray:
    """Compute normalized shortest-path distances on the undirected concept graph."""
    undirected = graph.to_undirected()
    n = graph.number_of_nodes()
    dist = np.zeros((n, n), dtype=float)
    lengths = dict(nx.all_pairs_shortest_path_length(undirected))
    max_len = max(max(v.values()) for v in lengths.values())
    max_len = max(max_len, 1)
    for i in range(n):
        for j in range(n):
            d = lengths[i][j] if j in lengths[i] else max_len
            dist[i, j] = d / max_len
    return dist


def prerequisite_coverage(mastery: np.ndarray, ancestors_map: Dict[int, List[int]], concept_idx: int, threshold: float) -> float:
    """Measure the fraction of prerequisite ancestors currently mastered."""
    prereqs = ancestors_map[concept_idx]
    if len(prereqs) == 0:
        return 1.0
    satisfied = [(mastery[p] >= threshold) for p in prereqs]
    return float(np.mean(satisfied))


def frontier_candidates(mastery: np.ndarray, ancestors_map: Dict[int, List[int]], threshold: float) -> List[int]:
    """Return concepts whose prerequisites are sufficiently satisfied but not yet mastered."""
    cands = []
    for concept_idx in range(len(mastery)):
        coverage = prerequisite_coverage(mastery, ancestors_map, concept_idx, threshold)
        if coverage >= 0.999 and mastery[concept_idx] < 0.88:
            cands.append(concept_idx)
    return cands


def review_candidates(mastery: np.ndarray, last_concept: int, dist_matrix: np.ndarray) -> List[int]:
    """Return concepts suitable for review near the learner's current region."""
    cands = []
    for concept_idx in range(len(mastery)):
        if mastery[concept_idx] < 0.72 and dist_matrix[last_concept, concept_idx] <= 0.45:
            cands.append(concept_idx)
    return cands


def exploratory_jump_candidates(mastery: np.ndarray, last_concept: int, dist_matrix: np.ndarray) -> List[int]:
    """Return higher-risk candidate concepts slightly ahead of the current frontier."""
    cands = []
    for concept_idx in range(len(mastery)):
        if mastery[concept_idx] < 0.65 and 0.10 <= dist_matrix[last_concept, concept_idx] <= 0.60:
            cands.append(concept_idx)
    return cands


def build_dynamic_learner_features(
    profile: str,
    support_need: float,
    perseverance: float,
    mastery: np.ndarray,
    frontier_size: int,
    recent_correctness: Sequence[float],
    recent_hints: Sequence[float],
) -> np.ndarray:
    """Encode the learner's dynamic state for the recommendation model.

    Features are intentionally bounded to [0, 1] so that the model can be used
    without external feature scaling when the script is adapted later.
    """
    recent_correct = safe_mean(recent_correctness, default=0.5)
    recent_help = clip01(safe_mean(recent_hints, default=0.0) / 4.0)
    mean_mastery = float(np.mean(mastery))
    frontier_norm = float(frontier_size / len(mastery))
    profile_vec = profile_one_hot(profile)
    out = np.concatenate(
        [
            profile_vec,
            np.array(
                [
                    float(support_need),
                    float(perseverance),
                    mean_mastery,
                    frontier_norm,
                    recent_correct,
                    recent_help,
                ],
                dtype=float,
            ),
        ]
    )
    return clip01(out)


def choose_resource_for_concept(
    resources_for_concept: pd.DataFrame,
    learner_prefs: np.ndarray,
    target_mastery: float,
    support_need: float,
    rng: np.random.Generator,
) -> pd.Series:
    """Choose a resource modality with context-sensitive probabilities."""
    weights = []
    for row in resources_for_concept.itertuples(index=False):
        pref_weight = float(learner_prefs[row.modality_idx])
        helpfulness = 1.0
        if row.modality in SUPPORTIVE_MODALITIES and target_mastery < 0.55:
            helpfulness += 0.65 + 0.30 * support_need
        if row.modality in ASSESSMENT_MODALITIES and target_mastery > 0.60:
            helpfulness += 0.30
        if row.modality == "quiz" and target_mastery < 0.40:
            helpfulness -= 0.35
        weights.append(max(0.05, pref_weight * helpfulness))
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    idx = int(rng.choice(np.arange(len(resources_for_concept)), p=weights))
    return resources_for_concept.iloc[idx]


def success_probability(
    mastery_before: float,
    prereq_cover: float,
    modality_match: float,
    difficulty: float,
    support_need: float,
    perseverance: float,
    concept_distance: float,
    modality: str,
    recent_correctness: Sequence[float],
    shifted: bool = False,
) -> float:
    """Probability of successful interaction in the synthetic environment.

    The equation intentionally balances learner state, curricular coherence,
    and resource modality effects so that topology-aware policies have a real
    signal to exploit.
    """
    recent_perf = safe_mean(recent_correctness, default=0.5)
    support_bonus = 0.0
    if modality in SUPPORTIVE_MODALITIES and mastery_before < 0.55:
        support_bonus += 0.45 + 0.25 * support_need
    if modality in ASSESSMENT_MODALITIES and mastery_before > 0.65:
        support_bonus += 0.18

    shift_penalty = 0.0
    if shifted:
        shift_penalty = 0.10 if modality in {"quiz", "practice"} else 0.03

    logit = (
        -0.90
        + 2.10 * mastery_before
        + 1.35 * prereq_cover
        + 0.55 * modality_match
        + 0.40 * perseverance
        + 0.25 * recent_perf
        - 1.55 * difficulty
        - 0.55 * concept_distance
        - shift_penalty
        + support_bonus
    )
    return float(expit(logit))


def update_mastery(
    mastery: np.ndarray,
    concept_idx: int,
    direct_prereqs: Dict[int, List[int]],
    graph: nx.DiGraph,
    correctness: int,
    modality: str,
    prereq_cover: float,
) -> np.ndarray:
    """Update mastery after an interaction.

    The update rule is deliberately simple but pedagogically interpretable:
    successful practice improves target mastery, support resources provide smaller
    but safer gains, and prerequisite alignment modulates transfer.
    """
    new_mastery = mastery.copy()
    if correctness == 1:
        gain = 0.08
        if modality in SUPPORTIVE_MODALITIES:
            gain += 0.02
        if modality in ASSESSMENT_MODALITIES:
            gain += 0.03
        gain += 0.03 * prereq_cover
    else:
        gain = 0.01 if modality in SUPPORTIVE_MODALITIES else -0.01

    new_mastery[concept_idx] = clip01(new_mastery[concept_idx] + gain)

    # Small prerequisite reinforcement: revisiting an advanced concept can also
    # strengthen prerequisite traces when success occurs.
    if correctness == 1:
        for p in direct_prereqs[concept_idx]:
            new_mastery[p] = clip01(new_mastery[p] + 0.012)
        for succ in graph.successors(concept_idx):
            new_mastery[succ] = clip01(new_mastery[succ] + 0.006)

    return new_mastery


def simulate_interaction_logs(
    specs: List[LearnerSpec],
    resources: pd.DataFrame,
    concept_graph: nx.DiGraph,
    ancestors_map: Dict[int, List[int]],
    direct_prereqs: Dict[int, List[int]],
    dist_matrix: np.ndarray,
    config: ExperimentConfig,
    rng: np.random.Generator,
    shifted: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic sequential educational logs.

    Returns
    -------
    interactions_df:
        Event-level sequential dataset.
    learners_df:
        Static learner summary table with aggregated features.
    """
    resources_by_concept = {
        c: resources.loc[resources["concept_idx"] == c].reset_index(drop=True)
        for c in resources["concept_idx"].unique()
    }

    interaction_rows: List[Dict[str, Any]] = []
    learner_summary_rows: List[Dict[str, Any]] = []

    for spec in specs:
        mastery = spec.initial_mastery.copy()
        last_concept = 0
        n_steps = int(rng.integers(config.min_interactions, config.max_interactions + 1))
        current_time = pd.Timestamp("2025-01-01") + pd.Timedelta(int(spec.learner_idx), unit="D")
        recent_correctness: deque[float] = deque(maxlen=4)
        recent_hints: deque[float] = deque(maxlen=4)
        recent_concepts: deque[int] = deque(maxlen=3)
        recent_gain_trace: deque[float] = deque(maxlen=3)

        total_correct = 0
        total_hints = 0.0
        total_duration = 0.0

        for step in range(n_steps):
            frontier = frontier_candidates(mastery, ancestors_map, config.mastery_threshold)
            review = review_candidates(mastery, last_concept, dist_matrix)
            jumps = exploratory_jump_candidates(mastery, last_concept, dist_matrix)

            # Mode probabilities encode the tension between curricular coherence and drift.
            if spec.profile == "underprepared":
                p_frontier, p_review, p_jump = 0.42, 0.45, 0.13
            elif spec.profile == "advanced":
                p_frontier, p_review, p_jump = 0.60, 0.20, 0.20
            else:
                p_frontier, p_review, p_jump = 0.52, 0.30, 0.18

            # When the learner struggles, the process becomes more review-heavy.
            if len(recent_correctness) > 0 and safe_mean(recent_correctness, 0.5) < 0.45:
                p_review += 0.15
                p_frontier -= 0.10
                p_jump -= 0.05

            p = np.array([p_frontier, p_review, p_jump], dtype=float)
            p = p / p.sum()
            mode = rng.choice(["frontier", "review", "jump"], p=p)

            if mode == "frontier" and frontier:
                scores = np.array([1.0 - mastery[c] for c in frontier], dtype=float)
                scores = scores / scores.sum()
                concept_idx = int(rng.choice(frontier, p=scores))
            elif mode == "review" and review:
                scores = np.array([1.0 - mastery[c] + 0.10 for c in review], dtype=float)
                scores = scores / scores.sum()
                concept_idx = int(rng.choice(review, p=scores))
            elif jumps:
                scores = np.array([1.0 - mastery[c] + 0.10 for c in jumps], dtype=float)
                scores = scores / scores.sum()
                concept_idx = int(rng.choice(jumps, p=scores))
            else:
                concept_idx = int(rng.integers(0, len(mastery)))

            mastery_snapshot = mastery.copy()
            prereq_cover = prerequisite_coverage(mastery_snapshot, ancestors_map, concept_idx, config.mastery_threshold)
            concept_distance = float(dist_matrix[last_concept, concept_idx])
            target_mastery = float(mastery_snapshot[concept_idx])

            res_row = choose_resource_for_concept(
                resources_by_concept[concept_idx],
                learner_prefs=spec.modality_preferences,
                target_mastery=target_mastery,
                support_need=spec.support_need,
                rng=rng,
            )

            modality_match = float(spec.modality_preferences[int(res_row.modality_idx)])
            p_success = success_probability(
                mastery_before=target_mastery,
                prereq_cover=prereq_cover,
                modality_match=modality_match,
                difficulty=float(res_row.difficulty),
                support_need=spec.support_need,
                perseverance=spec.perseverance,
                concept_distance=concept_distance,
                modality=str(res_row.modality),
                recent_correctness=list(recent_correctness),
                shifted=shifted,
            )
            correctness = int(rng.random() < p_success)
            score = float(clip01(0.40 + 0.60 * p_success + rng.normal(0.0, 0.08)) * 100.0)
            expected_duration = float(res_row.expected_duration)
            dwell_time = float(max(3.0, expected_duration * (0.8 + 0.7 * (1.0 - p_success) + rng.normal(0.0, 0.10))))
            hints_used = int(max(0, np.round(4.5 * (1.0 - p_success) + 2.0 * spec.support_need + rng.normal(0.0, 0.7))))
            effort = float(min(1.0, dwell_time / max(expected_duration, 1.0)))

            dynamic_features = build_dynamic_learner_features(
                profile=spec.profile,
                support_need=spec.support_need,
                perseverance=spec.perseverance,
                mastery=mastery_snapshot,
                frontier_size=len(frontier),
                recent_correctness=list(recent_correctness),
                recent_hints=list(recent_hints),
            )

            mastery_after = update_mastery(
                mastery_snapshot,
                concept_idx=concept_idx,
                direct_prereqs=direct_prereqs,
                graph=concept_graph,
                correctness=correctness,
                modality=str(res_row.modality),
                prereq_cover=prereq_cover,
            )
            mastery_gain = float(np.mean(mastery_after) - np.mean(mastery_snapshot))

            repeated_loop_flag = int(len(recent_concepts) >= 2 and list(recent_concepts)[-2:] == [concept_idx, concept_idx] and mastery_gain < 0.008)

            interaction_rows.append(
                {
                    "learner_idx": spec.learner_idx,
                    "learner_id": spec.learner_id,
                    "timestamp": current_time,
                    "step_idx": step,
                    "profile": spec.profile,
                    "support_need": spec.support_need,
                    "perseverance": spec.perseverance,
                    "concept_idx": concept_idx,
                    "last_concept_idx": last_concept,
                    "resource_idx": int(res_row.resource_idx),
                    "resource_id": str(res_row.resource_id),
                    "modality": str(res_row.modality),
                    "modality_idx": int(res_row.modality_idx),
                    "difficulty": float(res_row.difficulty),
                    "supportive_flag": float(res_row.supportive_flag),
                    "target_mastery": target_mastery,
                    "prereq_coverage": prereq_cover,
                    "concept_distance": concept_distance,
                    "modality_match": modality_match,
                    "correctness": correctness,
                    "score": score,
                    "hints_used": hints_used,
                    "dwell_time": dwell_time,
                    "effort": effort,
                    "success_probability": p_success,
                    "mastery_vector": mastery_snapshot.astype(float),
                    "modality_pref_vector": spec.modality_preferences.astype(float),
                    "learner_features": dynamic_features.astype(float),
                    "repeated_loop_flag": repeated_loop_flag,
                }
            )

            mastery = mastery_after
            total_correct += correctness
            total_hints += hints_used
            total_duration += dwell_time
            last_concept = concept_idx
            recent_correctness.append(float(correctness))
            recent_hints.append(float(hints_used))
            recent_concepts.append(concept_idx)
            recent_gain_trace.append(mastery_gain)
            current_time += pd.Timedelta(int(rng.integers(1, 4)), unit="D")

        learner_summary_rows.append(
            {
                "learner_idx": spec.learner_idx,
                "learner_id": spec.learner_id,
                "profile": spec.profile,
                "support_need": spec.support_need,
                "perseverance": spec.perseverance,
                "mean_correctness": total_correct / max(n_steps, 1),
                "mean_hints": total_hints / max(n_steps, 1),
                "mean_duration": total_duration / max(n_steps, 1),
                "final_mean_mastery": float(np.mean(mastery)),
                "final_mastery_std": float(np.std(mastery)),
                "interaction_count": n_steps,
                "modality_pref_vector": spec.modality_preferences.astype(float),
                "static_features": np.concatenate(
                    [
                        profile_one_hot(spec.profile),
                        np.array(
                            [
                                spec.support_need,
                                spec.perseverance,
                                float(np.mean(mastery)),
                                float(np.std(mastery)),
                                total_correct / max(n_steps, 1),
                                clip01((total_hints / max(n_steps, 1)) / 4.0),
                            ],
                            dtype=float,
                        ),
                    ]
                ).astype(float),
            }
        )

    interactions = pd.DataFrame(interaction_rows).sort_values(["learner_idx", "timestamp", "step_idx"]).reset_index(drop=True)
    learners_df = pd.DataFrame(learner_summary_rows).sort_values("learner_idx").reset_index(drop=True)
    return interactions, learners_df


# -----------------------------------------------------------------------------
# Splitting and supervised sample construction
# -----------------------------------------------------------------------------
def temporal_split(interactions: pd.DataFrame) -> pd.DataFrame:
    """Create train/validation/test labels by preserving temporal order per learner."""
    rows = []
    for _, grp in interactions.groupby("learner_idx", sort=False):
        grp = grp.sort_values(["timestamp", "step_idx"]).copy()
        n = len(grp)
        n_test = max(2, int(math.ceil(0.20 * n)))
        n_val = max(2, int(math.ceil(0.15 * (n - n_test))))
        split = np.array(["train"] * n, dtype=object)
        split[-n_test:] = "test"
        split[-(n_test + n_val):-n_test] = "val"
        grp["split"] = split
        rows.append(grp)
    return pd.concat(rows, ignore_index=True)


def build_example_row(
    learner_idx: int,
    item_idx: int,
    concept_idx: int,
    modality_idx: int,
    learner_features: np.ndarray,
    side_features: np.ndarray,
    label: int,
) -> Dict[str, Any]:
    """Create a single supervised example for PyTorch training."""
    return {
        "user_idx": int(learner_idx),
        "item_idx": int(item_idx),
        "concept_idx": int(concept_idx),
        "modality_idx": int(modality_idx),
        "learner_features": learner_features.astype(np.float32),
        "side_features": side_features.astype(np.float32),
        "label": float(label),
    }


def side_features_for_candidate(
    mastery_vector: np.ndarray,
    ancestors_map: Dict[int, List[int]],
    concept_idx: int,
    last_concept_idx: int,
    dist_matrix: np.ndarray,
    difficulty: float,
    modality_match: float,
    supportive_flag: float,
    mastery_threshold: float,
) -> np.ndarray:
    """Build candidate-level features used by the recommendation model."""
    return np.array(
        [
            float(mastery_vector[concept_idx]),
            prerequisite_coverage(mastery_vector, ancestors_map, concept_idx, mastery_threshold),
            float(dist_matrix[last_concept_idx, concept_idx]),
            float(difficulty),
            float(modality_match),
            float(supportive_flag),
        ],
        dtype=float,
    )


def create_supervised_examples(
    interactions: pd.DataFrame,
    resources: pd.DataFrame,
    learners_df: pd.DataFrame,
    ancestors_map: Dict[int, List[int]],
    dist_matrix: np.ndarray,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> Dict[str, List[Dict[str, Any]]]:
    """Convert sequential logs into positive/negative supervised examples.

    Positive examples correspond to historical learner-resource interactions.
    Negative examples are sampled from resources not selected at the same state.
    The sampler intentionally mixes near-miss negatives and random negatives.
    """
    resources_by_idx = resources.set_index("resource_idx")
    learner_prefs = learners_df.set_index("learner_idx")["modality_pref_vector"].to_dict()

    split_examples: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for row in interactions.itertuples(index=False):
        mastery_vec = np.asarray(row.mastery_vector, dtype=float)
        learner_feats = np.asarray(row.learner_features, dtype=float)

        positive_side = side_features_for_candidate(
            mastery_vector=mastery_vec,
            ancestors_map=ancestors_map,
            concept_idx=int(row.concept_idx),
            last_concept_idx=int(row.last_concept_idx),
            dist_matrix=dist_matrix,
            difficulty=float(row.difficulty),
            modality_match=float(row.modality_match),
            supportive_flag=float(row.supportive_flag),
            mastery_threshold=config.mastery_threshold,
        )
        split_examples[row.split].append(
            build_example_row(
                learner_idx=int(row.learner_idx),
                item_idx=int(row.resource_idx),
                concept_idx=int(row.concept_idx),
                modality_idx=int(row.modality_idx),
                learner_features=learner_feats,
                side_features=positive_side,
                label=1,
            )
        )

        # Hard negatives: same concept / nearby concepts but different item.
        all_items = resources["resource_idx"].to_numpy()
        same_concept = resources.loc[(resources["concept_idx"] == int(row.concept_idx)) & (resources["resource_idx"] != int(row.resource_idx)), "resource_idx"].tolist()
        near_concepts = [
            c for c in range(dist_matrix.shape[0])
            if 0.0 < dist_matrix[int(row.last_concept_idx), c] <= 0.50 and c != int(row.concept_idx)
        ]
        nearby_items = resources.loc[resources["concept_idx"].isin(near_concepts), "resource_idx"].tolist()

        neg_pool = list(dict.fromkeys(same_concept + nearby_items))
        if len(neg_pool) < config.negative_ratio:
            remaining = [int(x) for x in all_items if int(x) != int(row.resource_idx) and int(x) not in neg_pool]
            rng.shuffle(remaining)
            neg_pool.extend(remaining)

        neg_items = neg_pool[: config.negative_ratio]
        for neg_item in neg_items:
            item_meta = resources_by_idx.loc[int(neg_item)]
            prefs = np.asarray(learner_prefs[int(row.learner_idx)], dtype=float)
            neg_side = side_features_for_candidate(
                mastery_vector=mastery_vec,
                ancestors_map=ancestors_map,
                concept_idx=int(item_meta.concept_idx),
                last_concept_idx=int(row.last_concept_idx),
                dist_matrix=dist_matrix,
                difficulty=float(item_meta.difficulty),
                modality_match=float(prefs[int(item_meta.modality_idx)]),
                supportive_flag=float(item_meta.supportive_flag),
                mastery_threshold=config.mastery_threshold,
            )
            split_examples[row.split].append(
                build_example_row(
                    learner_idx=int(row.learner_idx),
                    item_idx=int(item_meta.name),
                    concept_idx=int(item_meta.concept_idx),
                    modality_idx=int(item_meta.modality_idx),
                    learner_features=learner_feats,
                    side_features=neg_side,
                    label=0,
                )
            )

    return split_examples


# -----------------------------------------------------------------------------
# Graph construction for structural modeling and regularization
# -----------------------------------------------------------------------------
def build_resource_graph(interactions: pd.DataFrame, resources: pd.DataFrame) -> nx.Graph:
    """Construct a resource graph from sequential transitions and concept co-membership."""
    graph = nx.Graph()
    for row in resources.itertuples(index=False):
        graph.add_node(
            int(row.resource_idx),
            resource_name=row.resource_name,
            modality=row.modality,
            strand=row.strand,
            concept_idx=int(row.concept_idx),
        )

    # Sequential transition counts across learners
    transition_counter: Counter[Tuple[int, int]] = Counter()
    for _, grp in interactions.groupby("learner_idx", sort=False):
        res_seq = grp.sort_values(["timestamp", "step_idx"])["resource_idx"].tolist()
        for a, b in zip(res_seq[:-1], res_seq[1:]):
            if a == b:
                continue
            edge = tuple(sorted((int(a), int(b))))
            transition_counter[edge] += 1

    for (a, b), count in transition_counter.items():
        graph.add_edge(a, b, weight=float(count))

    # Same-concept and same-strand affinity helps stabilize sparse logs.
    grouped = resources.groupby("concept_idx")["resource_idx"].apply(list).to_dict()
    for item_list in grouped.values():
        for i in range(len(item_list)):
            for j in range(i + 1, len(item_list)):
                a, b = int(item_list[i]), int(item_list[j])
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1.5
                else:
                    graph.add_edge(a, b, weight=1.5)

    return graph


def build_learner_graph(learners_df: pd.DataFrame, k: int = 8) -> nx.Graph:
    """Build a learner similarity graph from aggregated learner attributes."""
    X = np.vstack(learners_df["static_features"].to_numpy())
    X = StandardScaler().fit_transform(X)
    nn_model = NearestNeighbors(n_neighbors=min(k + 1, len(learners_df))).fit(X)
    dists, idxs = nn_model.kneighbors(X)

    graph = nx.Graph()
    for row in learners_df.itertuples(index=False):
        graph.add_node(int(row.learner_idx), profile=row.profile)

    sigma = np.median(dists[:, 1:]) + 1e-6
    for i in range(len(learners_df)):
        for dist, j in zip(dists[i, 1:], idxs[i, 1:]):
            if i == j:
                continue
            weight = float(np.exp(-(dist ** 2) / (2.0 * sigma ** 2)))
            if graph.has_edge(i, int(j)):
                graph[i][int(j)]["weight"] = max(graph[i][int(j)]["weight"], weight)
            else:
                graph.add_edge(i, int(j), weight=weight)
    return graph


def build_concept_similarity_graph(concept_graph: nx.DiGraph) -> nx.Graph:
    """Convert the prerequisite graph into a weighted undirected smoothness graph."""
    g = nx.Graph()
    for node, attrs in concept_graph.nodes(data=True):
        g.add_node(node, **attrs)
    for u, v in concept_graph.edges():
        g.add_edge(u, v, weight=1.0)
    return g


def graph_to_edge_tensors(graph: nx.Graph) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a weighted NetworkX graph to edge-index tensors for PyTorch regularization."""
    edges = []
    weights = []
    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 1.0))
        edges.append((int(u), int(v)))
        weights.append(w)
    if len(edges) == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=DEVICE), torch.zeros((0,), dtype=torch.float32, device=DEVICE)
    return (
        torch.tensor(edges, dtype=torch.long, device=DEVICE),
        torch.tensor(weights, dtype=torch.float32, device=DEVICE),
    )


# -----------------------------------------------------------------------------
# PyTorch dataset and model
# -----------------------------------------------------------------------------
class InteractionDataset(Dataset):
    """Thin PyTorch dataset wrapper around the supervised examples."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.user_idx = torch.tensor([r["user_idx"] for r in rows], dtype=torch.long)
        self.item_idx = torch.tensor([r["item_idx"] for r in rows], dtype=torch.long)
        self.concept_idx = torch.tensor([r["concept_idx"] for r in rows], dtype=torch.long)
        self.modality_idx = torch.tensor([r["modality_idx"] for r in rows], dtype=torch.long)
        self.learner_features = torch.tensor(np.vstack([r["learner_features"] for r in rows]), dtype=torch.float32)
        self.side_features = torch.tensor(np.vstack([r["side_features"] for r in rows]), dtype=torch.float32)
        self.labels = torch.tensor([r["label"] for r in rows], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "user_idx": self.user_idx[idx],
            "item_idx": self.item_idx[idx],
            "concept_idx": self.concept_idx[idx],
            "modality_idx": self.modality_idx[idx],
            "learner_features": self.learner_features[idx],
            "side_features": self.side_features[idx],
            "label": self.labels[idx],
        }


class EducationalTopoRecommender(nn.Module):
    """Recommendation model with explicit capacity for topology-aware regularization.

    Architecture rationale
    ----------------------
    The model uses both identity-based learner embeddings and dynamic learner-state
    features. This choice is important for two reasons:

    1. It lets the model represent known learners from historical logs.
    2. It preserves a usable pathway for cold-start / rollout learners through an
       out-of-vocabulary learner embedding plus feature-based encoding.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_concepts: int,
        n_modalities: int,
        learner_feature_dim: int,
        side_feature_dim: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.n_users_known = n_users
        self.oov_user_index = n_users  # last slot reserved for rollouts / unseen learners

        self.user_embedding = nn.Embedding(n_users + 1, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.concept_embedding = nn.Embedding(n_concepts, embedding_dim)
        self.modality_embedding = nn.Embedding(n_modalities, embedding_dim)

        self.learner_encoder = nn.Sequential(
            nn.Linear(learner_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.side_encoder = nn.Sequential(
            nn.Linear(side_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, embedding_dim),
        )

        fusion_dim = embedding_dim * 9 + side_feature_dim
        self.scoring_mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-style initialization to keep training stable on CPU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for emb in [self.user_embedding, self.item_embedding, self.concept_embedding, self.modality_embedding]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.08)

    def encode_user(self, user_idx: torch.Tensor, learner_features: torch.Tensor) -> torch.Tensor:
        """Combine ID-based and feature-based learner representations."""
        return self.user_embedding(user_idx) + self.learner_encoder(learner_features)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        concept_idx: torch.Tensor,
        modality_idx: torch.Tensor,
        learner_features: torch.Tensor,
        side_features: torch.Tensor,
    ) -> torch.Tensor:
        user_vec = self.encode_user(user_idx, learner_features)
        item_vec = self.item_embedding(item_idx)
        concept_vec = self.concept_embedding(concept_idx)
        modality_vec = self.modality_embedding(modality_idx)
        side_vec = self.side_encoder(side_features)

        # Pairwise multiplicative interactions encode compatibility in latent space.
        cross = torch.cat(
            [
                user_vec,
                item_vec,
                concept_vec,
                modality_vec,
                side_vec,
                user_vec * item_vec,
                item_vec * concept_vec,
                user_vec * concept_vec,
                item_vec * modality_vec,
                side_features,
            ],
            dim=1,
        )
        return self.scoring_mlp(cross).squeeze(-1)


# -----------------------------------------------------------------------------
# Losses and training
# -----------------------------------------------------------------------------
def graph_smoothness_loss(embeddings: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    """Weighted Laplacian smoothness term sum w_ij ||z_i - z_j||^2."""
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    diff = embeddings[src] - embeddings[dst]
    sq = (diff * diff).sum(dim=1)
    return torch.mean(edge_weight * sq)


@dataclass
class TopologyRegularizationContext:
    learner_edges: torch.Tensor
    learner_weights: torch.Tensor
    resource_edges: torch.Tensor
    resource_weights: torch.Tensor
    concept_edges: torch.Tensor
    concept_weights: torch.Tensor
    static_learner_features: torch.Tensor
    item_to_concept: torch.Tensor


@dataclass
class TrainingResult:
    model: EducationalTopoRecommender
    history: pd.DataFrame


@torch.no_grad()
def predict_binary_scores(model: EducationalTopoRecommender, dataset: InteractionDataset) -> np.ndarray:
    """Compute logits for all rows in a dataset."""
    model.eval()
    logits = model(
        dataset.user_idx.to(DEVICE),
        dataset.item_idx.to(DEVICE),
        dataset.concept_idx.to(DEVICE),
        dataset.modality_idx.to(DEVICE),
        dataset.learner_features.to(DEVICE),
        dataset.side_features.to(DEVICE),
    )
    return logits.detach().cpu().numpy()


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Return robust binary metrics for model monitoring during training."""
    probs = expit(logits)
    if len(np.unique(labels)) < 2:
        return {"auc": float("nan"), "ap": float("nan")}
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "ap": float(average_precision_score(labels, probs)),
    }


def train_recommender(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    n_users: int,
    n_items: int,
    n_concepts: int,
    n_modalities: int,
    reg_context: TopologyRegularizationContext,
    config: ExperimentConfig,
    topology_aware: bool,
) -> TrainingResult:
    """Train either the baseline or the topology-aware recommender."""
    train_ds = InteractionDataset(train_rows)
    val_ds = InteractionDataset(val_rows)
    loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    learner_feature_dim = train_ds.learner_features.shape[1]
    side_feature_dim = train_ds.side_features.shape[1]
    model = EducationalTopoRecommender(
        n_users=n_users,
        n_items=n_items,
        n_concepts=n_concepts,
        n_modalities=n_modalities,
        learner_feature_dim=learner_feature_dim,
        side_feature_dim=side_feature_dim,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    history_rows: List[Dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_task = 0.0
        epoch_total = 0.0
        batch_count = 0

        for batch in loader:
            optimizer.zero_grad()
            user_idx = batch["user_idx"].to(DEVICE)
            item_idx = batch["item_idx"].to(DEVICE)
            concept_idx = batch["concept_idx"].to(DEVICE)
            modality_idx = batch["modality_idx"].to(DEVICE)
            learner_features = batch["learner_features"].to(DEVICE)
            side_features = batch["side_features"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(user_idx, item_idx, concept_idx, modality_idx, learner_features, side_features)
            task_loss = criterion(logits, labels)
            total_loss = task_loss

            learner_smooth = torch.tensor(0.0, device=DEVICE)
            resource_smooth = torch.tensor(0.0, device=DEVICE)
            concept_smooth = torch.tensor(0.0, device=DEVICE)
            alignment = torch.tensor(0.0, device=DEVICE)
            prereq_pen = torch.tensor(0.0, device=DEVICE)
            stability_pen = torch.tensor(0.0, device=DEVICE)

            if topology_aware:
                # Learner smoothness is applied to full learner representations, not merely ID embeddings.
                known_user_idx = torch.arange(n_users, device=DEVICE)
                known_user_repr = model.encode_user(known_user_idx, reg_context.static_learner_features)
                learner_smooth = graph_smoothness_loss(known_user_repr, reg_context.learner_edges, reg_context.learner_weights)
                resource_smooth = graph_smoothness_loss(model.item_embedding.weight, reg_context.resource_edges, reg_context.resource_weights)
                concept_smooth = graph_smoothness_loss(model.concept_embedding.weight, reg_context.concept_edges, reg_context.concept_weights)

                aligned_concepts = model.concept_embedding(reg_context.item_to_concept)
                alignment = torch.mean((model.item_embedding.weight - aligned_concepts) ** 2)

                probs = torch.sigmoid(logits)
                prereq_gap = 1.0 - side_features[:, 1]  # 1 - prerequisite coverage
                step_size = side_features[:, 2]
                supportive_mask = side_features[:, 5]
                prereq_pen = torch.mean(probs * prereq_gap * (0.50 + step_size) * (1.0 - supportive_mask))

                learner_noise = torch.clamp(learner_features + 0.03 * torch.randn_like(learner_features), 0.0, 1.0)
                side_noise = torch.clamp(side_features + 0.03 * torch.randn_like(side_features), 0.0, 1.0)
                perturbed_logits = model(user_idx, item_idx, concept_idx, modality_idx, learner_noise, side_noise)
                stability_pen = torch.mean((torch.sigmoid(logits) - torch.sigmoid(perturbed_logits)) ** 2)

                total_loss = (
                    task_loss
                    + config.lambda_learner_smooth * learner_smooth
                    + config.lambda_resource_smooth * resource_smooth
                    + config.lambda_concept_smooth * concept_smooth
                    + config.lambda_alignment * alignment
                    + config.lambda_prereq * prereq_pen
                    + config.lambda_stability * stability_pen
                )

            total_loss.backward()
            optimizer.step()

            epoch_task += float(task_loss.item())
            epoch_total += float(total_loss.item())
            batch_count += 1

        val_logits = predict_binary_scores(model, val_ds)
        val_labels = val_ds.labels.numpy()
        cls = classification_metrics(val_logits, val_labels)

        history_rows.append(
            {
                "epoch": epoch,
                "model": "topology_aware" if topology_aware else "baseline",
                "train_task_loss": epoch_task / max(batch_count, 1),
                "train_total_loss": epoch_total / max(batch_count, 1),
                "val_auc": cls["auc"],
                "val_ap": cls["ap"],
                "learner_smooth": float(learner_smooth.detach().cpu().item()),
                "resource_smooth": float(resource_smooth.detach().cpu().item()),
                "concept_smooth": float(concept_smooth.detach().cpu().item()),
                "alignment": float(alignment.detach().cpu().item()),
                "prereq_penalty": float(prereq_pen.detach().cpu().item()),
                "stability_penalty": float(stability_pen.detach().cpu().item()),
            }
        )

    return TrainingResult(model=model, history=pd.DataFrame(history_rows))


# -----------------------------------------------------------------------------
# Recommendation scoring and ranking evaluation
# -----------------------------------------------------------------------------
def score_all_resources_for_state(
    model: EducationalTopoRecommender,
    user_idx: int,
    learner_features: np.ndarray,
    mastery_vector: np.ndarray,
    last_concept_idx: int,
    modality_preferences: np.ndarray,
    resources: pd.DataFrame,
    ancestors_map: Dict[int, List[int]],
    dist_matrix: np.ndarray,
    mastery_threshold: float,
) -> pd.DataFrame:
    """Score the entire resource catalog for a single learner state."""
    n = len(resources)
    concept_idx = resources["concept_idx"].to_numpy(dtype=int)
    modality_idx = resources["modality_idx"].to_numpy(dtype=int)
    difficulty = resources["difficulty"].to_numpy(dtype=float)
    supportive = resources["supportive_flag"].to_numpy(dtype=float)
    modality_match = modality_preferences[modality_idx]

    side = np.vstack(
        [
            np.array(
                [
                    mastery_vector[c],
                    prerequisite_coverage(mastery_vector, ancestors_map, int(c), mastery_threshold),
                    dist_matrix[int(last_concept_idx), int(c)],
                    d,
                    modality_match[idx],
                    supportive[idx],
                ],
                dtype=float,
            )
            for idx, (c, d) in enumerate(zip(concept_idx, difficulty))
        ]
    )
    learner_feat_batch = np.repeat(learner_features[None, :], n, axis=0)
    user_idx_batch = np.full(n, user_idx, dtype=np.int64)

    with torch.no_grad():
        logits = model(
            torch.tensor(user_idx_batch, dtype=torch.long, device=DEVICE),
            torch.tensor(resources["resource_idx"].to_numpy(dtype=np.int64), dtype=torch.long, device=DEVICE),
            torch.tensor(concept_idx, dtype=torch.long, device=DEVICE),
            torch.tensor(modality_idx, dtype=torch.long, device=DEVICE),
            torch.tensor(learner_feat_batch, dtype=torch.float32, device=DEVICE),
            torch.tensor(side, dtype=torch.float32, device=DEVICE),
        ).cpu().numpy()

    out = resources[["resource_idx", "resource_name", "concept_idx", "concept_name", "modality", "supportive_flag", "difficulty"]].copy()
    out["score"] = logits
    out["probability"] = expit(logits)
    out["target_mastery"] = side[:, 0]
    out["prereq_coverage"] = side[:, 1]
    out["concept_distance"] = side[:, 2]
    out["modality_match"] = side[:, 4]
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def apply_topology_constraints(
    ranked_df: pd.DataFrame,
    profile: str,
    mean_mastery: float,
    k: int,
) -> pd.DataFrame:
    """Rerank by enforcing prerequisite consistency and bounded concept step size.

    This function implements the chapter's idea of a safety envelope. Advanced
    learners are allowed slightly larger curricular jumps, while underprepared
    learners are held to tighter reachability constraints unless a support-oriented
    resource is being recommended.
    """
    if profile == "underprepared":
        max_step = 0.35 if mean_mastery < 0.60 else 0.45
        min_coverage = 0.85 if mean_mastery < 0.60 else 0.70
    elif profile == "advanced":
        max_step = 0.60 if mean_mastery < 0.70 else 0.75
        min_coverage = 0.60
    else:
        max_step = 0.50 if mean_mastery < 0.65 else 0.60
        min_coverage = 0.70

    work = ranked_df.copy()
    allowed = (
        (work["prereq_coverage"] >= min_coverage) & (work["concept_distance"] <= max_step)
    ) | (
        (work["supportive_flag"] >= 0.5) & (work["concept_distance"] <= max_step + 0.12)
    )

    filtered = work.loc[allowed].copy()

    # If the safety filter is too strict, relax it gradually rather than failing.
    relaxation = 0
    while len(filtered) < k and relaxation < 3:
        relaxation += 1
        step_relax = 0.08 * relaxation
        cov_relax = 0.10 * relaxation
        allowed = (
            (work["prereq_coverage"] >= max(0.40, min_coverage - cov_relax))
            & (work["concept_distance"] <= min(1.0, max_step + step_relax))
        ) | (
            (work["supportive_flag"] >= 0.5)
            & (work["concept_distance"] <= min(1.0, max_step + step_relax + 0.12))
        )
        filtered = work.loc[allowed].copy()

    if len(filtered) == 0:
        filtered = work.copy()

    filtered = filtered.sort_values(["score", "prereq_coverage"], ascending=[False, False]).reset_index(drop=True)
    return filtered


def topk_jaccard(original: Sequence[int], perturbed: Sequence[int]) -> float:
    """Compute top-k set stability via Jaccard overlap."""
    a = set(int(x) for x in original)
    b = set(int(x) for x in perturbed)
    if len(a | b) == 0:
        return 1.0
    return float(len(a & b) / len(a | b))


def compute_event_metrics(
    ranking: pd.DataFrame,
    actual_resource_idx: int,
    k: int,
) -> Dict[str, float]:
    """Compute one-event ranking and coherence metrics."""
    ranking = ranking.reset_index(drop=True)
    topk = ranking.head(k)
    ranks = ranking.index[ranking["resource_idx"] == int(actual_resource_idx)].tolist()
    rank = int(ranks[0] + 1) if ranks else 10_000
    recall = float(rank <= k)
    ndcg = float(1.0 / math.log2(rank + 1)) if rank <= k else 0.0
    mrr = float(1.0 / rank) if rank <= k else 0.0
    violation = float(np.mean((topk["prereq_coverage"] < 0.75) & (topk["supportive_flag"] < 0.5)))
    step = float(np.mean(topk["concept_distance"]))
    support_share = float(np.mean(topk["supportive_flag"]))
    return {
        "recall_at_k": recall,
        "ndcg_at_k": ndcg,
        "mrr_at_k": mrr,
        "prerequisite_violation_rate": violation,
        "mean_step_size": step,
        "support_share": support_share,
    }


def evaluate_offline_rankings(
    baseline_model: EducationalTopoRecommender,
    topo_model: EducationalTopoRecommender,
    test_events: pd.DataFrame,
    resources: pd.DataFrame,
    learners_df: pd.DataFrame,
    ancestors_map: Dict[int, List[int]],
    dist_matrix: np.ndarray,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate baseline, topology-aware, and full constrained policies offline."""
    learner_meta = learners_df.set_index("learner_idx")
    policy_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []

    for row in test_events.itertuples(index=False):
        modality_prefs = np.asarray(row.modality_pref_vector, dtype=float)
        learner_features = np.asarray(row.learner_features, dtype=float)
        mastery_vector = np.asarray(row.mastery_vector, dtype=float)
        learner_profile = str(row.profile)
        mean_mastery = float(np.mean(mastery_vector))

        scored_baseline = score_all_resources_for_state(
            baseline_model,
            user_idx=int(row.learner_idx),
            learner_features=learner_features,
            mastery_vector=mastery_vector,
            last_concept_idx=int(row.last_concept_idx),
            modality_preferences=modality_prefs,
            resources=resources,
            ancestors_map=ancestors_map,
            dist_matrix=dist_matrix,
            mastery_threshold=config.mastery_threshold,
        )
        scored_topo = score_all_resources_for_state(
            topo_model,
            user_idx=int(row.learner_idx),
            learner_features=learner_features,
            mastery_vector=mastery_vector,
            last_concept_idx=int(row.last_concept_idx),
            modality_preferences=modality_prefs,
            resources=resources,
            ancestors_map=ancestors_map,
            dist_matrix=dist_matrix,
            mastery_threshold=config.mastery_threshold,
        )
        scored_topo_constrained = apply_topology_constraints(scored_topo, learner_profile, mean_mastery, config.top_k)

        # Stability check: slightly perturb the learner state and recompute top-k lists.
        perturbed_mastery = clip01(mastery_vector + rng.normal(0.0, 0.03, size=len(mastery_vector)))
        perturbed_features = clip01(learner_features + rng.normal(0.0, 0.02, size=len(learner_features)))
        pert_topo = score_all_resources_for_state(
            topo_model,
            user_idx=int(row.learner_idx),
            learner_features=perturbed_features,
            mastery_vector=perturbed_mastery,
            last_concept_idx=int(row.last_concept_idx),
            modality_preferences=modality_prefs,
            resources=resources,
            ancestors_map=ancestors_map,
            dist_matrix=dist_matrix,
            mastery_threshold=config.mastery_threshold,
        )

        policy_dict = {
            "baseline_plain": scored_baseline,
            "topology_plain": scored_topo,
            "topology_constrained": scored_topo_constrained,
        }

        for policy_name, ranking in policy_dict.items():
            metrics = compute_event_metrics(ranking, actual_resource_idx=int(row.resource_idx), k=config.top_k)
            if policy_name == "baseline_plain":
                pert_ref = score_all_resources_for_state(
                    baseline_model,
                    user_idx=int(row.learner_idx),
                    learner_features=perturbed_features,
                    mastery_vector=perturbed_mastery,
                    last_concept_idx=int(row.last_concept_idx),
                    modality_preferences=modality_prefs,
                    resources=resources,
                    ancestors_map=ancestors_map,
                    dist_matrix=dist_matrix,
                    mastery_threshold=config.mastery_threshold,
                )
                stability = topk_jaccard(ranking.head(config.top_k)["resource_idx"], pert_ref.head(config.top_k)["resource_idx"])
            else:
                # The constrained policy is compared to the perturbed topological model.
                pert_use = pert_topo if policy_name == "topology_plain" else apply_topology_constraints(pert_topo, learner_profile, float(np.mean(perturbed_mastery)), config.top_k)
                stability = topk_jaccard(ranking.head(config.top_k)["resource_idx"], pert_use.head(config.top_k)["resource_idx"])

            out = {
                "policy": policy_name,
                "learner_idx": int(row.learner_idx),
                "profile": learner_profile,
                "actual_resource_idx": int(row.resource_idx),
                "actual_concept_idx": int(row.concept_idx),
                "stability_jaccard": stability,
                **metrics,
            }
            event_rows.append(out)

    event_df = pd.DataFrame(event_rows)
    summary = event_df.groupby("policy", as_index=False).agg(
        recall_at_k=("recall_at_k", "mean"),
        ndcg_at_k=("ndcg_at_k", "mean"),
        mrr_at_k=("mrr_at_k", "mean"),
        prerequisite_violation_rate=("prerequisite_violation_rate", "mean"),
        mean_step_size=("mean_step_size", "mean"),
        support_share=("support_share", "mean"),
        stability_jaccard=("stability_jaccard", "mean"),
    )
    return summary, event_df


# -----------------------------------------------------------------------------
# Persistent homology and latent topology summaries
# -----------------------------------------------------------------------------
def sample_balanced_indices(labels: Sequence[str], max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Sample indices in a roughly profile-balanced way for topology plots."""
    labels = np.asarray(labels)
    unique = np.unique(labels)
    per_group = max(1, max_points // max(len(unique), 1))
    chosen = []
    for group in unique:
        idx = np.where(labels == group)[0]
        if len(idx) == 0:
            continue
        take = min(len(idx), per_group)
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    if len(chosen) < min(max_points, len(labels)):
        remaining = [i for i in range(len(labels)) if i not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: min(max_points, len(labels)) - len(chosen)])
    return np.array(sorted(chosen), dtype=int)


def persistence_summary(diagrams: List[np.ndarray]) -> Dict[str, float]:
    """Summarize persistence diagrams for H0 and H1."""
    out: Dict[str, float] = {}
    for dim, dgm in enumerate(diagrams[:2]):
        if dgm is None or len(dgm) == 0:
            out[f"H{dim}_count"] = 0.0
            out[f"H{dim}_total_lifetime"] = 0.0
            out[f"H{dim}_max_lifetime"] = 0.0
            continue
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            out[f"H{dim}_count"] = float(len(dgm))
            out[f"H{dim}_total_lifetime"] = 0.0
            out[f"H{dim}_max_lifetime"] = 0.0
            continue
        lifetimes = finite[:, 1] - finite[:, 0]
        out[f"H{dim}_count"] = float(len(dgm))
        out[f"H{dim}_total_lifetime"] = float(np.sum(lifetimes))
        out[f"H{dim}_max_lifetime"] = float(np.max(lifetimes))
    return out


def betti_curve_from_diagram(diagram: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Compute a Betti curve directly from a persistence diagram."""
    if diagram is None or len(diagram) == 0:
        return np.zeros_like(grid)
    finite = diagram[np.isfinite(diagram[:, 1])]
    if len(finite) == 0:
        return np.zeros_like(grid)
    curve = np.zeros_like(grid)
    for i, t in enumerate(grid):
        curve[i] = np.sum((finite[:, 0] <= t) & (finite[:, 1] > t))
    return curve


def compute_latent_representations(
    model: EducationalTopoRecommender,
    learners_df: pd.DataFrame,
) -> np.ndarray:
    """Get one latent representation per learner using static learner features."""
    user_idx = torch.tensor(learners_df["learner_idx"].to_numpy(dtype=np.int64), dtype=torch.long, device=DEVICE)
    static_feat = torch.tensor(np.vstack(learners_df["static_features"].to_numpy()), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        z = model.encode_user(user_idx, static_feat).cpu().numpy()
    return z


# -----------------------------------------------------------------------------
# Higher-order topology with TopoNetX
# -----------------------------------------------------------------------------
def build_learning_simplicial_complex(interactions: pd.DataFrame, concept_graph: nx.DiGraph) -> Any:
    """Lift concept sequences to a simplicial complex using TopoNetX.

    Sliding windows of learner concept sequences create higher-order relations.
    In practice this approximates recurrent local learning neighborhoods and
    review cycles across concepts.
    """
    sc = tnx.SimplicialComplex()

    for node in concept_graph.nodes():
        sc.add_simplex([int(node)])
    for u, v in concept_graph.to_undirected().edges():
        sc.add_simplex([int(u), int(v)])

    for _, grp in interactions.groupby("learner_idx", sort=False):
        seq = grp.sort_values(["timestamp", "step_idx"])["concept_idx"].tolist()
        for start in range(len(seq) - 2):
            simplex = tuple(sorted(set(seq[start:start + 3])))
            if len(simplex) >= 2:
                sc.add_simplex(simplex)

    return sc


def summarize_simplicial_complex(sc: Any) -> pd.DataFrame:
    """Produce a compact summary of the higher-order learning complex."""
    shape = tuple(int(x) for x in sc.shape)
    hodge_1 = sc.hodge_laplacian_matrix(rank=1)
    if hasattr(hodge_1, "toarray"):
        hodge_arr = hodge_1.toarray()
    else:
        hodge_arr = np.asarray(hodge_1)
    evals = np.linalg.eigvalsh(hodge_arr) if hodge_arr.size > 0 else np.array([])
    harmonic_count = int(np.sum(np.isclose(evals, 0.0, atol=1e-8))) if len(evals) > 0 else 0
    smallest_positive = float(np.min(evals[evals > 1e-8])) if np.any(evals > 1e-8) else 0.0

    rows = [
        {
            "n_0_simplices": shape[0] if len(shape) > 0 else 0,
            "n_1_simplices": shape[1] if len(shape) > 1 else 0,
            "n_2_simplices": shape[2] if len(shape) > 2 else 0,
            "complex_dimension": int(sc.dim),
            "harmonic_1_cycles": harmonic_count,
            "smallest_positive_hodge_eigenvalue": smallest_positive,
        }
    ]
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Mapper graph for governance/auditing maps
# -----------------------------------------------------------------------------
def build_concept_feature_table(
    interactions: pd.DataFrame,
    concept_graph: nx.DiGraph,
    resources: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate concept-level features for Mapper visualization."""
    usage = interactions.groupby("concept_idx").size().rename("usage")
    success = interactions.groupby("concept_idx")["correctness"].mean().rename("success_rate")
    failure = (1.0 - success).rename("failure_rate")
    hints = interactions.groupby("concept_idx")["hints_used"].mean().rename("mean_hints")
    dwell = interactions.groupby("concept_idx")["dwell_time"].mean().rename("mean_dwell_time")
    diff = resources.groupby("concept_idx")["difficulty"].mean().rename("mean_difficulty")

    pagerank = pd.Series(nx.pagerank(concept_graph.to_undirected()), name="pagerank")
    betweenness = pd.Series(nx.betweenness_centrality(concept_graph.to_undirected()), name="betweenness")
    clustering = pd.Series(nx.clustering(concept_graph.to_undirected()), name="local_clustering")
    depth = pd.Series({n: concept_graph.nodes[n]["depth"] for n in concept_graph.nodes()}, name="depth")
    out_degree = pd.Series(dict(concept_graph.out_degree()), name="out_degree")
    in_degree = pd.Series(dict(concept_graph.in_degree()), name="in_degree")

    df = pd.concat([usage, success, failure, hints, dwell, diff, pagerank, betweenness, clustering, depth, out_degree, in_degree], axis=1).reset_index()
    df = df.rename(columns={"index": "concept_idx"})
    df = df.fillna(0.0)
    df["priority_score"] = (
        0.20 * df["usage"] / max(df["usage"].max(), 1.0)
        + 0.15 * df["failure_rate"]
        + 0.20 * df["betweenness"]
        + 0.15 * df["pagerank"]
        + 0.15 * (df["depth"] / max(df["depth"].max(), 1.0))
        + 0.15 * df["mean_hints"] / max(df["mean_hints"].max(), 1.0)
    )
    df["concept_name"] = df["concept_idx"].map({n: concept_graph.nodes[n]["name"] for n in concept_graph.nodes()})
    return df


def build_mapper_graph(concept_features: pd.DataFrame, config: ExperimentConfig) -> Tuple[nx.Graph, Dict[str, Any]]:
    """Build a Mapper graph using KeplerMapper and return a NetworkX view."""
    X = concept_features[
        [
            "usage",
            "success_rate",
            "failure_rate",
            "mean_hints",
            "mean_dwell_time",
            "mean_difficulty",
            "pagerank",
            "betweenness",
            "local_clustering",
            "priority_score",
        ]
    ].to_numpy(dtype=float)

    X = StandardScaler().fit_transform(X)
    mapper = km.KeplerMapper(verbose=0)
    lens = mapper.fit_transform(X, projection=PCA(n_components=2, random_state=config.random_state))
    cover = km.Cover(n_cubes=config.mapper_n_cubes, perc_overlap=config.mapper_perc_overlap)

    if len(X) < 5:
        clusterer = DBSCAN(eps=0.5, min_samples=2)
    else:
        nn_model = NearestNeighbors(n_neighbors=min(3, len(X))).fit(X)
        dists, _ = nn_model.kneighbors(X)
        eps = float(max(0.15, np.quantile(dists[:, -1], 0.75)))
        clusterer = DBSCAN(eps=eps, min_samples=config.mapper_dbscan_min_samples)

    graph = mapper.map(lens, X, clusterer=clusterer, cover=cover)

    nx_graph = nx.Graph()
    for node_id, members in graph["nodes"].items():
        nx_graph.add_node(node_id, size=len(members), members=members)
    for src, dst_list in graph["links"].items():
        for dst in dst_list:
            nx_graph.add_edge(src, dst)

    return nx_graph, graph


# -----------------------------------------------------------------------------
# Rollout evaluation for personalized tutoring policies
# -----------------------------------------------------------------------------
def observed_mastery_under_scenario(
    true_mastery: np.ndarray,
    history: List[np.ndarray],
    scenario: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Transform true mastery into the observed mastery used by the recommender."""
    if scenario == "iid":
        return true_mastery.copy()
    if scenario == "new_semester":
        return clip01(true_mastery + rng.normal(0.0, 0.04, size=len(true_mastery)))
    if scenario == "missingness":
        mask = rng.random(len(true_mastery)) > 0.22
        observed = true_mastery.copy()
        observed[~mask] = 0.50
        observed = clip01(observed + rng.normal(0.0, 0.03, size=len(true_mastery)))
        return observed
    if scenario == "delay":
        if len(history) >= 3:
            return history[-3].copy()
        return history[0].copy()
    raise ValueError(f"Unknown scenario: {scenario}")


def scenario_adjusted_difficulty(base_difficulty: float, concept_idx: int, scenario: str) -> float:
    """Apply scenario-specific difficulty shifts during rollouts."""
    if scenario == "new_semester":
        if concept_idx >= 5:
            return float(clip01(base_difficulty + 0.08))
    return float(base_difficulty)


def simulate_policy_rollouts(
    baseline_model: EducationalTopoRecommender,
    topo_model: EducationalTopoRecommender,
    resources: pd.DataFrame,
    concept_graph: nx.DiGraph,
    ancestors_map: Dict[int, List[int]],
    direct_prereqs: Dict[int, List[int]],
    dist_matrix: np.ndarray,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run short tutoring episodes under multiple robustness scenarios."""
    rollout_specs = generate_learner_specs(config.n_learners_rollout, concept_graph.number_of_nodes(), rng)
    results_rows: List[Dict[str, Any]] = []
    trajectory_rows: List[Dict[str, Any]] = []

    policies = ["baseline_plain", "topology_constrained"]
    scenarios = ["iid", "new_semester", "missingness", "delay"]

    for scenario in scenarios:
        for spec in rollout_specs:
            for policy_name in policies:
                true_mastery = spec.initial_mastery.copy()
                observed_history = [true_mastery.copy()]
                recent_correctness: deque[float] = deque(maxlen=4)
                recent_hints: deque[float] = deque(maxlen=4)
                recent_concepts: deque[int] = deque(maxlen=3)
                last_concept = 0
                cumulative_gain = 0.0
                success_count = 0
                violation_count = 0
                step_sizes: List[float] = []
                loop_count = 0

                for step in range(config.rollout_steps):
                    observed_mastery = observed_mastery_under_scenario(true_mastery, observed_history, scenario, rng)
                    front = frontier_candidates(observed_mastery, ancestors_map, config.mastery_threshold)
                    learner_features = build_dynamic_learner_features(
                        profile=spec.profile,
                        support_need=spec.support_need,
                        perseverance=spec.perseverance,
                        mastery=observed_mastery,
                        frontier_size=len(front),
                        recent_correctness=list(recent_correctness),
                        recent_hints=list(recent_hints),
                    )

                    model = baseline_model if policy_name == "baseline_plain" else topo_model
                    user_idx = model.oov_user_index
                    scored = score_all_resources_for_state(
                        model,
                        user_idx=user_idx,
                        learner_features=learner_features,
                        mastery_vector=observed_mastery,
                        last_concept_idx=last_concept,
                        modality_preferences=spec.modality_preferences,
                        resources=resources,
                        ancestors_map=ancestors_map,
                        dist_matrix=dist_matrix,
                        mastery_threshold=config.mastery_threshold,
                    )

                    if policy_name == "topology_constrained":
                        scored = apply_topology_constraints(scored, spec.profile, float(np.mean(observed_mastery)), k=config.top_k)
                    choice = scored.iloc[0]

                    chosen_concept = int(choice.concept_idx)
                    chosen_modality = str(choice.modality)
                    prereq_cover = float(choice.prereq_coverage)
                    step_size = float(choice.concept_distance)
                    difficulty = scenario_adjusted_difficulty(float(choice.difficulty), chosen_concept, scenario)
                    modality_match = float(spec.modality_preferences[MODALITIES.index(chosen_modality)])
                    p_success = success_probability(
                        mastery_before=float(true_mastery[chosen_concept]),
                        prereq_cover=prereq_cover,
                        modality_match=modality_match,
                        difficulty=difficulty,
                        support_need=spec.support_need,
                        perseverance=spec.perseverance,
                        concept_distance=step_size,
                        modality=chosen_modality,
                        recent_correctness=list(recent_correctness),
                        shifted=(scenario != "iid"),
                    )
                    correctness = int(rng.random() < p_success)

                    before_mean = float(np.mean(true_mastery))
                    true_mastery = update_mastery(
                        true_mastery,
                        concept_idx=chosen_concept,
                        direct_prereqs=direct_prereqs,
                        graph=concept_graph,
                        correctness=correctness,
                        modality=chosen_modality,
                        prereq_cover=prereq_cover,
                    )
                    gain = float(np.mean(true_mastery) - before_mean)
                    cumulative_gain += gain
                    success_count += correctness
                    violation_count += int((prereq_cover < 0.75) and (choice.supportive_flag < 0.5))
                    step_sizes.append(step_size)
                    recent_correctness.append(float(correctness))
                    recent_hints.append(float(max(0, round(4.0 * (1.0 - p_success) + spec.support_need))))
                    if len(recent_concepts) >= 2 and list(recent_concepts)[-2:] == [chosen_concept, chosen_concept] and gain < 0.008:
                        loop_count += 1
                    recent_concepts.append(chosen_concept)
                    last_concept = chosen_concept
                    observed_history.append(true_mastery.copy())

                    if spec.learner_idx < 4:
                        trajectory_rows.append(
                            {
                                "scenario": scenario,
                                "policy": policy_name,
                                "learner_idx": spec.learner_idx,
                                "profile": spec.profile,
                                "step": step,
                                "chosen_concept": chosen_concept,
                                "chosen_modality": chosen_modality,
                                "prereq_coverage": prereq_cover,
                                "step_size": step_size,
                                "success": correctness,
                                "mean_mastery_after": float(np.mean(true_mastery)),
                            }
                        )

                results_rows.append(
                    {
                        "scenario": scenario,
                        "policy": policy_name,
                        "learner_idx": spec.learner_idx,
                        "profile": spec.profile,
                        "cumulative_learning_gain": cumulative_gain,
                        "final_mean_mastery": float(np.mean(true_mastery)),
                        "success_rate": success_count / config.rollout_steps,
                        "prerequisite_violation_rate": violation_count / config.rollout_steps,
                        "mean_step_size": safe_mean(step_sizes),
                        "loop_rate": loop_count / config.rollout_steps,
                    }
                )

    return pd.DataFrame(results_rows), pd.DataFrame(trajectory_rows)


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def plot_concept_graph(graph: nx.DiGraph, pos: Dict[int, Tuple[float, float]], path: Path) -> None:
    """Visualize the prerequisite graph."""
    plt.figure(figsize=(11, 4))
    node_colors = []
    strand_to_color = {
        "Foundations": "#4c78a8",
        "Algebra": "#f58518",
        "Modeling": "#54a24b",
        "Data": "#e45756",
        "Capstone": "#72b7b2",
    }
    for node in graph.nodes():
        node_colors.append(strand_to_color.get(graph.nodes[node]["strand"], "#999999"))
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle="-|>", width=1.5, alpha=0.7)
    nx.draw_networkx_nodes(graph, pos, node_size=950, node_color=node_colors, edgecolors="black", linewidths=0.8)
    labels = {n: graph.nodes[n]["name"].replace(" and ", "\n") for n in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    plt.title("Figure 5.1A. Curriculum prerequisite graph")
    plt.axis("off")
    save_figure(path)


def plot_resource_graph(resource_graph: nx.Graph, resources: pd.DataFrame, path: Path) -> None:
    """Visualize the resource transition / co-usage graph."""
    plt.figure(figsize=(9, 7))
    pos = nx.spring_layout(resource_graph, seed=11, k=0.40)
    modality_color = {
        "reading": "#4c78a8",
        "video": "#72b7b2",
        "practice": "#54a24b",
        "quiz": "#e45756",
        "tutoring": "#b279a2",
    }
    node_colors = [modality_color[resource_graph.nodes[n]["modality"]] for n in resource_graph.nodes()]
    edge_widths = [0.6 + 0.18 * resource_graph[u][v].get("weight", 1.0) for u, v in resource_graph.edges()]
    nx.draw_networkx_edges(resource_graph, pos, alpha=0.15, width=edge_widths)
    nx.draw_networkx_nodes(resource_graph, pos, node_size=90, node_color=node_colors, alpha=0.95, edgecolors="black", linewidths=0.2)
    plt.title("Figure 5.1B. Resource transition and co-usage graph")
    plt.axis("off")
    save_figure(path)


def plot_learner_state_cloud(learners_df: pd.DataFrame, path: Path) -> None:
    """Plot learner static feature clouds in a low-dimensional view."""
    X = np.vstack(learners_df["static_features"].to_numpy())
    xy = PCA(n_components=2, random_state=17).fit_transform(X)
    plt.figure(figsize=(7, 5))
    markers = {"underprepared": "o", "regular": "s", "advanced": "^"}
    for profile in PROFILE_ORDER:
        mask = learners_df["profile"] == profile
        plt.scatter(xy[mask, 0], xy[mask, 1], label=profile, alpha=0.75, s=45, marker=markers[profile])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Figure 5.1C. Learner state cloud by preparation profile")
    plt.legend(frameon=True)
    plt.grid(alpha=0.25)
    save_figure(path)


def plot_latent_spaces(baseline_z: np.ndarray, topo_z: np.ndarray, labels: Sequence[str], path: Path) -> None:
    """Compare baseline and topology-aware learner manifolds."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    coords_a = PCA(n_components=2, random_state=7).fit_transform(baseline_z)
    coords_b = PCA(n_components=2, random_state=7).fit_transform(topo_z)
    for ax, coords, title in zip(
        axes,
        [coords_a, coords_b],
        ["Baseline latent space", "Topology-aware latent space"],
    ):
        for profile in PROFILE_ORDER:
            mask = np.asarray(labels) == profile
            ax.scatter(coords[mask, 0], coords[mask, 1], label=profile, alpha=0.72, s=36)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=True)
    fig.suptitle("Figure 5.2A. Latent learner manifolds before and after topology-aware regularization")
    save_figure(path)


def plot_persistence_diagrams_comparison(dgms_a: List[np.ndarray], dgms_b: List[np.ndarray], path: Path) -> None:
    """Plot baseline vs topology-aware persistence diagrams."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_diagrams(dgms_a, title="Baseline latent topology", ax=axes[0], show=False)
    plot_diagrams(dgms_b, title="Topology-aware latent topology", ax=axes[1], show=False)
    fig.suptitle("Figure 5.2B. Persistent homology of learner latent representations")
    save_figure(path)


def plot_betti_curves(dgms_a: List[np.ndarray], dgms_b: List[np.ndarray], path: Path, grid_size: int = 160) -> None:
    """Plot Betti curves for H0 and H1 without requiring giotto-tda."""
    all_points = []
    for dgm in dgms_a[:2] + dgms_b[:2]:
        if dgm is not None and len(dgm) > 0:
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) > 0:
                all_points.extend(finite.ravel().tolist())
    max_t = max(all_points) if len(all_points) > 0 else 1.0
    grid = np.linspace(0.0, max_t, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for dim, ax in enumerate(axes):
        curve_a = betti_curve_from_diagram(dgms_a[dim], grid)
        curve_b = betti_curve_from_diagram(dgms_b[dim], grid)
        ax.plot(grid, curve_a, label="Baseline")
        ax.plot(grid, curve_b, label="Topology-aware")
        ax.set_title(f"Betti curve H{dim}")
        ax.set_xlabel("Filtration value")
        ax.set_ylabel("Betti number")
        ax.grid(alpha=0.25)
        ax.legend(frameon=True)
    fig.suptitle("Figure 5.2C. Betti curves of latent learner spaces")
    save_figure(path)


def plot_training_history(history_df: pd.DataFrame, path: Path) -> None:
    """Plot learning curves for baseline and topology-aware models."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for model_name, grp in history_df.groupby("model"):
        axes[0].plot(grp["epoch"], grp["train_task_loss"], label=model_name)
        axes[1].plot(grp["epoch"], grp["val_auc"], label=model_name)
        axes[2].plot(grp["epoch"], grp["prereq_penalty"], label=model_name)
    axes[0].set_title("Task loss")
    axes[1].set_title("Validation AUC")
    axes[2].set_title("Prerequisite penalty")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(frameon=True)
    fig.suptitle("Figure 5.2D. Training dynamics of baseline and topology-aware models")
    save_figure(path)


def plot_policy_comparison(summary_df: pd.DataFrame, path: Path) -> None:
    """Bar chart for offline policy comparison."""
    metrics = ["ndcg_at_k", "prerequisite_violation_rate", "mean_step_size", "stability_jaccard"]
    titles = ["NDCG@K", "Violation rate", "Mean step size", "Stability"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar(summary_df["policy"], summary_df[metric])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.20, axis="y")
    fig.suptitle("Figure 5.3A. Offline ranking quality, coherence, and stability by policy")
    save_figure(path)


def plot_rollout_robustness(rollout_df: pd.DataFrame, path: Path) -> None:
    """Compare cumulative gains and violation rates across scenarios."""
    agg = rollout_df.groupby(["scenario", "policy"], as_index=False).agg(
        cumulative_learning_gain=("cumulative_learning_gain", "mean"),
        prerequisite_violation_rate=("prerequisite_violation_rate", "mean"),
    )
    scenarios = ["iid", "new_semester", "missingness", "delay"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in zip(
        axes,
        ["cumulative_learning_gain", "prerequisite_violation_rate"],
        ["Cumulative learning gain", "Prerequisite violation rate"],
    ):
        width = 0.35
        x = np.arange(len(scenarios))
        for j, policy in enumerate(["baseline_plain", "topology_constrained"]):
            vals = [float(agg.loc[(agg["scenario"] == s) & (agg["policy"] == policy), metric].iloc[0]) for s in scenarios]
            ax.bar(x + (j - 0.5) * width, vals, width=width, label=policy)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(frameon=True)
    fig.suptitle("Figure 5.3B. Robustness of tutoring policies under scenario shift")
    save_figure(path)


def plot_subgroup_audit(rollout_df: pd.DataFrame, path: Path) -> None:
    """Audit policy behavior across learner preparation profiles."""
    agg = rollout_df.groupby(["profile", "policy"], as_index=False).agg(
        cumulative_learning_gain=("cumulative_learning_gain", "mean"),
        prerequisite_violation_rate=("prerequisite_violation_rate", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(PROFILE_ORDER))
    width = 0.35
    for ax, metric, title in zip(
        axes,
        ["cumulative_learning_gain", "prerequisite_violation_rate"],
        ["Cumulative learning gain by profile", "Violation rate by profile"],
    ):
        for j, policy in enumerate(["baseline_plain", "topology_constrained"]):
            vals = [float(agg.loc[(agg["profile"] == p) & (agg["policy"] == policy), metric].iloc[0]) for p in PROFILE_ORDER]
            ax.bar(x + (j - 0.5) * width, vals, width=width, label=policy)
        ax.set_xticks(x)
        ax.set_xticklabels(PROFILE_ORDER, rotation=20)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(frameon=True)
    fig.suptitle("Figure 5.3C. Subgroup audit across learner preparation profiles")
    save_figure(path)


def plot_mapper_graph(nx_graph: nx.Graph, path: Path) -> None:
    """Render a static image of the Mapper graph."""
    plt.figure(figsize=(7.5, 5.5))
    if len(nx_graph) == 0:
        plt.text(0.5, 0.5, "Mapper graph is empty under current clustering settings.", ha="center", va="center")
        plt.axis("off")
        save_figure(path)
        return
    pos = nx.spring_layout(nx_graph, seed=19)
    sizes = [200 + 120 * nx_graph.nodes[n].get("size", 1) for n in nx_graph.nodes()]
    nx.draw_networkx_edges(nx_graph, pos, alpha=0.25, width=1.2)
    nx.draw_networkx_nodes(nx_graph, pos, node_size=sizes, alpha=0.90)
    nx.draw_networkx_labels(nx_graph, pos, font_size=7)
    plt.title("Figure 5.3D. Mapper governance graph over concept-level educational features")
    plt.axis("off")
    save_figure(path)


def plot_simplicial_skeleton(sc: Any, concept_graph: nx.DiGraph, path: Path) -> None:
    """Plot the 1-skeleton of the learning simplicial complex."""
    g = sc.graph_skeleton()
    plt.figure(figsize=(8.5, 4.8))
    pos = nx.spring_layout(g, seed=29)
    labels = {n: concept_graph.nodes[int(n)]["name"] for n in g.nodes() if int(n) in concept_graph.nodes()}
    nx.draw_networkx_edges(g, pos, alpha=0.25, width=1.0)
    nx.draw_networkx_nodes(g, pos, node_size=420, alpha=0.95, edgecolors="black", linewidths=0.4)
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)
    plt.title("Figure 5.3E. 1-skeleton of the learning simplicial complex")
    plt.axis("off")
    save_figure(path)


def plot_example_trajectories(trajectory_df: pd.DataFrame, path: Path) -> None:
    """Plot a small sample of tutoring trajectories for manuscript illustration."""
    subset = trajectory_df.loc[trajectory_df["scenario"] == "iid"].copy()
    subset = subset.loc[subset["learner_idx"] < 4].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, learner_idx in zip(axes, sorted(subset["learner_idx"].unique())):
        for policy in ["baseline_plain", "topology_constrained"]:
            data = subset.loc[(subset["learner_idx"] == learner_idx) & (subset["policy"] == policy)]
            ax.plot(data["step"], data["mean_mastery_after"], marker="o", label=policy)
        profile = subset.loc[subset["learner_idx"] == learner_idx, "profile"].iloc[0]
        ax.set_title(f"Learner {learner_idx} ({profile})")
        ax.set_xlabel("Tutoring step")
        ax.set_ylabel("Mean mastery")
        ax.grid(alpha=0.25)
        ax.legend(frameon=True)
    fig.suptitle("Figure 5.3F. Example tutoring trajectories under two policies")
    save_figure(path)


# -----------------------------------------------------------------------------
# Main experiment orchestration
# -----------------------------------------------------------------------------
def run_experiment(config: ExperimentConfig) -> Dict[str, Path]:
    """Run the full pipeline and return the output paths."""
    check_dependencies()
    set_global_seed(config.random_state)
    rng = np.random.default_rng(config.random_state)
    paths = ensure_output_tree(config.output_dir)
    manifest = ArtifactLogger()

    # 1) Build the synthetic curriculum and data.
    concept_graph, concept_df, concept_pos = build_concept_graph()
    resources = build_resource_catalog(concept_df, rng)
    specs = generate_learner_specs(config.n_learners_train, concept_graph.number_of_nodes(), rng)
    direct_prereqs, ancestors_map = build_prerequisite_maps(concept_graph)
    dist_matrix = compute_concept_distance_matrix(concept_graph)

    interactions, learners_df = simulate_interaction_logs(
        specs=specs,
        resources=resources,
        concept_graph=concept_graph,
        ancestors_map=ancestors_map,
        direct_prereqs=direct_prereqs,
        dist_matrix=dist_matrix,
        config=config,
        rng=rng,
        shifted=False,
    )
    interactions = temporal_split(interactions)

    # Save shared data summaries for transparency and reproducibility.
    concept_df.to_csv(paths["shared_tab"] / "shared_concepts.csv", index=False)
    resources.to_csv(paths["shared_tab"] / "shared_resources.csv", index=False)
    learners_clean = learners_df.drop(columns=["modality_pref_vector", "static_features"])
    learners_clean.to_csv(paths["shared_tab"] / "shared_learners.csv", index=False)
    interactions_clean = interactions.drop(columns=["mastery_vector", "modality_pref_vector", "learner_features"])
    interactions_clean.to_csv(paths["shared_tab"] / "shared_interactions.csv", index=False)

    # 2) Build graphs for structural modeling and regularization.
    learner_graph = build_learner_graph(learners_df)
    resource_graph = build_resource_graph(interactions, resources)
    concept_smooth_graph = build_concept_similarity_graph(concept_graph)

    learner_edges, learner_weights = graph_to_edge_tensors(learner_graph)
    resource_edges, resource_weights = graph_to_edge_tensors(resource_graph)
    concept_edges, concept_weights = graph_to_edge_tensors(concept_smooth_graph)

    reg_context = TopologyRegularizationContext(
        learner_edges=learner_edges,
        learner_weights=learner_weights,
        resource_edges=resource_edges,
        resource_weights=resource_weights,
        concept_edges=concept_edges,
        concept_weights=concept_weights,
        static_learner_features=torch.tensor(np.vstack(learners_df["static_features"].to_numpy()), dtype=torch.float32, device=DEVICE),
        item_to_concept=torch.tensor(resources["concept_idx"].to_numpy(dtype=np.int64), dtype=torch.long, device=DEVICE),
    )

    # 3) Supervised examples for model training.
    split_examples = create_supervised_examples(
        interactions=interactions,
        resources=resources,
        learners_df=learners_df,
        ancestors_map=ancestors_map,
        dist_matrix=dist_matrix,
        config=config,
        rng=rng,
    )

    # 4) Train baseline and topology-aware models.
    baseline_result = train_recommender(
        train_rows=split_examples["train"],
        val_rows=split_examples["val"],
        n_users=len(learners_df),
        n_items=len(resources),
        n_concepts=len(concept_df),
        n_modalities=len(MODALITIES),
        reg_context=reg_context,
        config=config,
        topology_aware=False,
    )
    topo_result = train_recommender(
        train_rows=split_examples["train"],
        val_rows=split_examples["val"],
        n_users=len(learners_df),
        n_items=len(resources),
        n_concepts=len(concept_df),
        n_modalities=len(MODALITIES),
        reg_context=reg_context,
        config=config,
        topology_aware=True,
    )
    history_df = pd.concat([baseline_result.history, topo_result.history], ignore_index=True)

    # 5) Offline policy evaluation.
    test_events = interactions.loc[interactions["split"] == "test"].copy().reset_index(drop=True)
    offline_summary, offline_event_metrics = evaluate_offline_rankings(
        baseline_model=baseline_result.model,
        topo_model=topo_result.model,
        test_events=test_events,
        resources=resources,
        learners_df=learners_df,
        ancestors_map=ancestors_map,
        dist_matrix=dist_matrix,
        config=config,
        rng=rng,
    )

    # 6) Latent representations and persistent homology.
    baseline_z = compute_latent_representations(baseline_result.model, learners_df)
    topo_z = compute_latent_representations(topo_result.model, learners_df)
    sample_idx = sample_balanced_indices(learners_df["profile"].tolist(), min(config.ph_sample_size, len(learners_df)), rng)
    baseline_sample = baseline_z[sample_idx]
    topo_sample = topo_z[sample_idx]
    baseline_dgms = ripser(baseline_sample, maxdim=1)["dgms"]
    topo_dgms = ripser(topo_sample, maxdim=1)["dgms"]

    topo_landscaper = PersistenceLandscaper(hom_deg=1, start=0.0, stop=1.0, num_steps=200, flatten=True)
    try:
        landscape_baseline = topo_landscaper.fit_transform([baseline_dgms[1]])[0]
        landscape_topo = topo_landscaper.fit_transform([topo_dgms[1]])[0]
        landscape_distance = float(np.linalg.norm(landscape_baseline - landscape_topo))
    except Exception:
        landscape_distance = float("nan")

    ph_summary = pd.DataFrame(
        [
            {
                "model": "baseline",
                **persistence_summary(baseline_dgms),
            },
            {
                "model": "topology_aware",
                **persistence_summary(topo_dgms),
            },
        ]
    )
    ph_summary["wasserstein_H1_to_other_model"] = [
        float(wasserstein(baseline_dgms[1], topo_dgms[1])) if len(baseline_dgms) > 1 and len(topo_dgms) > 1 else float("nan"),
        float(wasserstein(topo_dgms[1], baseline_dgms[1])) if len(baseline_dgms) > 1 and len(topo_dgms) > 1 else float("nan"),
    ]
    ph_summary["landscape_distance_between_models"] = landscape_distance

    # 7) Higher-order topology and Mapper.
    simplicial_complex = build_learning_simplicial_complex(interactions, concept_graph)
    simplicial_summary = summarize_simplicial_complex(simplicial_complex)
    concept_feature_table = build_concept_feature_table(interactions, concept_graph, resources)
    mapper_nx, mapper_raw = build_mapper_graph(concept_feature_table, config)
    mapper_summary = pd.DataFrame(
        [
            {
                "n_nodes": int(mapper_nx.number_of_nodes()),
                "n_edges": int(mapper_nx.number_of_edges()),
                "n_connected_components": int(nx.number_connected_components(mapper_nx)) if mapper_nx.number_of_nodes() > 0 else 0,
                "largest_component_size": int(max((len(c) for c in nx.connected_components(mapper_nx)), default=0)),
                "mean_mapper_node_size": float(np.mean([mapper_nx.nodes[n].get("size", 1) for n in mapper_nx.nodes()])) if mapper_nx.number_of_nodes() > 0 else 0.0,
            }
        ]
    )

    # 8) Rollout evaluation for personalized tutoring.
    rollout_df, trajectory_df = simulate_policy_rollouts(
        baseline_model=baseline_result.model,
        topo_model=topo_result.model,
        resources=resources,
        concept_graph=concept_graph,
        ancestors_map=ancestors_map,
        direct_prereqs=direct_prereqs,
        dist_matrix=dist_matrix,
        config=config,
        rng=rng,
    )

    # ------------------------------------------------------------------
    # Section 5.1 outputs
    # ------------------------------------------------------------------
    fig_51a = paths["s51_fig"] / "figure_5_1a_concept_graph.png"
    plot_concept_graph(concept_graph, concept_pos, fig_51a)
    manifest.add(
        "5.1",
        "figure",
        fig_51a,
        "Curricular structure",
        "Figure 5.1A. Curriculum prerequisite graph.",
        "Use this figure to explain reachability, prerequisite chains, and why coherence matters in educational guidance.",
    )

    fig_51b = paths["s51_fig"] / "figure_5_1b_resource_graph.png"
    plot_resource_graph(resource_graph, resources, fig_51b)
    manifest.add(
        "5.1",
        "figure",
        fig_51b,
        "Resource topology",
        "Figure 5.1B. Resource transition and co-usage graph.",
        "Useful for discussing structural relations among readings, videos, practice items, quizzes, and tutoring interventions.",
    )

    fig_51c = paths["s51_fig"] / "figure_5_1c_learner_state_cloud.png"
    plot_learner_state_cloud(learners_df, fig_51c)
    manifest.add(
        "5.1",
        "figure",
        fig_51c,
        "Learner heterogeneity",
        "Figure 5.1C. Learner state cloud by preparation profile.",
        "Supports the argument that personalization operates over heterogeneous learner regions rather than a single homogeneous cohort.",
    )

    structural_summary = pd.DataFrame(
        [
            {
                "n_learners": len(learners_df),
                "n_resources": len(resources),
                "n_concepts": len(concept_df),
                "n_interactions": len(interactions),
                "mean_correctness": float(interactions["correctness"].mean()),
                "mean_prereq_coverage": float(interactions["prereq_coverage"].mean()),
                "mean_concept_distance": float(interactions["concept_distance"].mean()),
                "resource_graph_nodes": int(resource_graph.number_of_nodes()),
                "resource_graph_edges": int(resource_graph.number_of_edges()),
                "learner_graph_nodes": int(learner_graph.number_of_nodes()),
                "learner_graph_edges": int(learner_graph.number_of_edges()),
            }
        ]
    )
    tab_51 = paths["s51_tab"] / "table_5_1_structural_summary.csv"
    structural_summary.to_csv(tab_51, index=False)
    manifest.add(
        "5.1",
        "table",
        tab_51,
        "Structural summary",
        "Table 5.1. Structural summary of the synthetic educational environment.",
        "This table can anchor the description of dataset scale, graph structure, and average curricular coherence before modeling.",
    )

    # ------------------------------------------------------------------
    # Section 5.2 outputs
    # ------------------------------------------------------------------
    fig_52a = paths["s52_fig"] / "figure_5_2a_latent_spaces.png"
    plot_latent_spaces(baseline_z, topo_z, learners_df["profile"].tolist(), fig_52a)
    manifest.add(
        "5.2",
        "figure",
        fig_52a,
        "Latent manifold comparison",
        "Figure 5.2A. Latent learner manifolds before and after topology-aware regularization.",
        "Use this to discuss whether the regularized representation is smoother, less fragmented, and more profile-coherent.",
    )

    fig_52b = paths["s52_fig"] / "figure_5_2b_persistence_diagrams.png"
    plot_persistence_diagrams_comparison(baseline_dgms, topo_dgms, fig_52b)
    manifest.add(
        "5.2",
        "figure",
        fig_52b,
        "Persistent homology",
        "Figure 5.2B. Persistent homology of learner latent representations.",
        "This figure supports analysis of fragmentation (H0) and cycle structure (H1) across baseline and regularized latent spaces.",
    )

    fig_52c = paths["s52_fig"] / "figure_5_2c_betti_curves.png"
    plot_betti_curves(baseline_dgms, topo_dgms, fig_52c, grid_size=config.betti_grid_size)
    manifest.add(
        "5.2",
        "figure",
        fig_52c,
        "Betti curves",
        "Figure 5.2C. Betti curves of baseline and topology-aware learner spaces.",
        "Use this to explain multiscale topological persistence in a way that is more accessible than raw persistence diagrams.",
    )

    fig_52d = paths["s52_fig"] / "figure_5_2d_training_history.png"
    plot_training_history(history_df, fig_52d)
    manifest.add(
        "5.2",
        "figure",
        fig_52d,
        "Training dynamics",
        "Figure 5.2D. Training dynamics of baseline and topology-aware models.",
        "This figure helps show that the topological penalties are not purely decorative; they actively reshape learning dynamics.",
    )

    model_summary = offline_summary.copy()
    model_summary["mean_H0_total_lifetime"] = [
        ph_summary.loc[ph_summary["model"] == "baseline", "H0_total_lifetime"].iloc[0],
        ph_summary.loc[ph_summary["model"] == "topology_aware", "H0_total_lifetime"].iloc[0],
        ph_summary.loc[ph_summary["model"] == "topology_aware", "H0_total_lifetime"].iloc[0],
    ]
    tab_52a = paths["s52_tab"] / "table_5_2_model_summary.csv"
    model_summary.to_csv(tab_52a, index=False)
    manifest.add(
        "5.2",
        "table",
        tab_52a,
        "Model comparison",
        "Table 5.2A. Offline model and policy comparison.",
        "This is the central table for discussing trade-offs between ranking utility, stability, and prerequisite coherence.",
    )

    tab_52b = paths["s52_tab"] / "table_5_2_persistence_summary.csv"
    ph_summary.to_csv(tab_52b, index=False)
    manifest.add(
        "5.2",
        "table",
        tab_52b,
        "Topological summary",
        "Table 5.2B. Persistent homology summary of latent spaces.",
        "Use this table to report H0/H1 counts, lifetimes, and inter-model diagram distances.",
    )

    tab_52c = paths["s52_tab"] / "table_5_2_training_history.csv"
    history_df.to_csv(tab_52c, index=False)
    manifest.add(
        "5.2",
        "table",
        tab_52c,
        "Training log",
        "Table 5.2C. Epoch-level training history.",
        "Useful if the manuscript needs a finer-grained technical appendix on optimization behavior.",
    )

    # ------------------------------------------------------------------
    # Section 5.3 outputs
    # ------------------------------------------------------------------
    fig_53a = paths["s53_fig"] / "figure_5_3a_offline_policy_comparison.png"
    plot_policy_comparison(offline_summary, fig_53a)
    manifest.add(
        "5.3",
        "figure",
        fig_53a,
        "Offline policy audit",
        "Figure 5.3A. Offline ranking quality, coherence, and stability by policy.",
        "This figure supports the claim that the full topology-aware pipeline changes decision quality, not just latent geometry.",
    )

    fig_53b = paths["s53_fig"] / "figure_5_3b_rollout_robustness.png"
    plot_rollout_robustness(rollout_df, fig_53b)
    manifest.add(
        "5.3",
        "figure",
        fig_53b,
        "Robustness under shift",
        "Figure 5.3B. Robustness of tutoring policies under scenario shift.",
        "Use this figure to argue that topology-aware tutoring remains more coherent under missingness, delay, and new-semester shift.",
    )

    fig_53c = paths["s53_fig"] / "figure_5_3c_subgroup_audit.png"
    plot_subgroup_audit(rollout_df, fig_53c)
    manifest.add(
        "5.3",
        "figure",
        fig_53c,
        "Subgroup audit",
        "Figure 5.3C. Subgroup audit across learner preparation profiles.",
        "This figure supports a governance-oriented discussion of whether structural benefits are shared across learner groups.",
    )

    fig_53d = paths["s53_fig"] / "figure_5_3d_mapper_graph.png"
    plot_mapper_graph(mapper_nx, fig_53d)
    manifest.add(
        "5.3",
        "figure",
        fig_53d,
        "Mapper governance map",
        "Figure 5.3D. Mapper governance graph over concept-level educational features.",
        "Use this figure to discuss how concept regions cluster topologically according to usage, difficulty, failure, and centrality patterns.",
    )

    fig_53e = paths["s53_fig"] / "figure_5_3e_simplicial_skeleton.png"
    plot_simplicial_skeleton(simplicial_complex, concept_graph, fig_53e)
    manifest.add(
        "5.3",
        "figure",
        fig_53e,
        "Higher-order learning structure",
        "Figure 5.3E. 1-skeleton of the learning simplicial complex.",
        "This figure can support discussion of higher-order learning neighborhoods beyond pairwise edges.",
    )

    fig_53f = paths["s53_fig"] / "figure_5_3f_example_trajectories.png"
    plot_example_trajectories(trajectory_df, fig_53f)
    manifest.add(
        "5.3",
        "figure",
        fig_53f,
        "Tutoring trajectories",
        "Figure 5.3F. Example tutoring trajectories under two policies.",
        "Use this to narrate how topology-constrained tutoring avoids abrupt curricular jumps and accumulates mastery more smoothly.",
    )

    # KeplerMapper interactive HTML export.
    mapper_html = paths["s53"] / "figure_5_3d_mapper_graph_interactive.html"
    try:
        mapper_obj = km.KeplerMapper(verbose=0)
        mapper_obj.visualize(
            mapper_raw,
            path_html=str(mapper_html),
            title="Educational concept Mapper graph",
            color_values=concept_feature_table["priority_score"].to_numpy(dtype=float),
            color_function_name="priority_score",
            save_file=True,
        )
        manifest.add(
            "5.3",
            "interactive_html",
            mapper_html,
            "Interactive Mapper graph",
            "Interactive Mapper graph for exploratory analysis.",
            "This is not mandatory for the manuscript PDF, but it is useful while deciding which regions to describe in the text.",
        )
    except Exception as exc:
        warnings.warn(f"Mapper HTML export failed: {exc}")

    tab_53a = paths["s53_tab"] / "table_5_3_rollout_summary.csv"
    rollout_df.groupby(["scenario", "policy"], as_index=False).agg(
        cumulative_learning_gain=("cumulative_learning_gain", "mean"),
        final_mean_mastery=("final_mean_mastery", "mean"),
        success_rate=("success_rate", "mean"),
        prerequisite_violation_rate=("prerequisite_violation_rate", "mean"),
        mean_step_size=("mean_step_size", "mean"),
        loop_rate=("loop_rate", "mean"),
    ).to_csv(tab_53a, index=False)
    manifest.add(
        "5.3",
        "table",
        tab_53a,
        "Rollout summary",
        "Table 5.3A. Sequential tutoring outcomes under multiple robustness scenarios.",
        "This is the main sequential-decision table for discussing tutoring performance and safety envelopes.",
    )

    tab_53b = paths["s53_tab"] / "table_5_3_subgroup_audit.csv"
    rollout_df.groupby(["profile", "policy"], as_index=False).agg(
        cumulative_learning_gain=("cumulative_learning_gain", "mean"),
        final_mean_mastery=("final_mean_mastery", "mean"),
        success_rate=("success_rate", "mean"),
        prerequisite_violation_rate=("prerequisite_violation_rate", "mean"),
        mean_step_size=("mean_step_size", "mean"),
        loop_rate=("loop_rate", "mean"),
    ).to_csv(tab_53b, index=False)
    manifest.add(
        "5.3",
        "table",
        tab_53b,
        "Subgroup audit table",
        "Table 5.3B. Subgroup audit across learner preparation profiles.",
        "Use this table when discussing equity, robustness, and differential policy behavior by learner group.",
    )

    tab_53c = paths["s53_tab"] / "table_5_3_simplicial_summary.csv"
    simplicial_summary.to_csv(tab_53c, index=False)
    manifest.add(
        "5.3",
        "table",
        tab_53c,
        "Higher-order topology summary",
        "Table 5.3C. Simplicial-complex summary of concept-sequence topology.",
        "This table supports discussion of higher-order learning relations, harmonic cycles, and structural organization beyond graphs.",
    )

    tab_53d = paths["s53_tab"] / "table_5_3_mapper_summary.csv"
    mapper_summary.to_csv(tab_53d, index=False)
    manifest.add(
        "5.3",
        "table",
        tab_53d,
        "Mapper summary",
        "Table 5.3D. Summary of the Mapper governance graph.",
        "Use this table to report the size and connectedness of the Mapper representation before interpreting its regions.",
    )

    tab_53e = paths["s53_tab"] / "table_5_3_example_trajectories.csv"
    trajectory_df.to_csv(tab_53e, index=False)
    manifest.add(
        "5.3",
        "table",
        tab_53e,
        "Example trajectories",
        "Table 5.3E. Step-level tutoring trajectories for selected learners.",
        "Useful as an appendix-style complement to the trajectory figure when the manuscript discusses individual learner pathways.",
    )

    # Final manifest and configuration files.
    manifest_df = manifest.to_dataframe()
    manifest_path = paths["root"] / "artifact_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    with open(paths["root"] / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    final_summary = {
        "baseline_val_auc_last": float(baseline_result.history["val_auc"].iloc[-1]),
        "topology_val_auc_last": float(topo_result.history["val_auc"].iloc[-1]),
        "offline_policy_summary_file": str(tab_52a),
        "rollout_summary_file": str(tab_53a),
        "manifest_file": str(manifest_path),
    }
    with open(paths["root"] / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    return paths


# -----------------------------------------------------------------------------
# Command-line entry point
# -----------------------------------------------------------------------------
def parse_args() -> ExperimentConfig:
    """Parse command-line arguments into an ExperimentConfig."""
    parser = argparse.ArgumentParser(description="Topological regularization in educational recommendation and tutoring")
    parser.add_argument("--output_dir", type=str, default="results_topological_education", help="Root directory for all outputs.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for full reproducibility.")
    parser.add_argument("--epochs", type=int, default=28, help="Number of training epochs for each model.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for PyTorch training.")
    parser.add_argument("--n_learners_train", type=int, default=180, help="Number of synthetic learners in the historical training cohort.")
    parser.add_argument("--n_learners_rollout", type=int, default=72, help="Number of synthetic learners used in sequential rollout evaluation.")
    args = parser.parse_args()
    cfg = ExperimentConfig(
        output_dir=args.output_dir,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_learners_train=args.n_learners_train,
        n_learners_rollout=args.n_learners_rollout,
    )
    return cfg


def main() -> None:
    """Execute the full experiment."""
    config = parse_args()
    run_experiment(config)


if __name__ == "__main__":
    main()
