
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Topological pipeline for Chapter 5 of
"Topological Regularization in Educational Recommendation and Personalized Tutoring".

This script is designed to generate a compact, publication-oriented set of
figures and tables for the three target sections of the chapter:

    5.1 Expected gains in coherence, stability, and personalization
    5.2 Topological regularization in deep learning models
    5.3 Topology-based educational recommendation and governance maps

Main design goals
-----------------
1. Keep runtime moderate:
   - compact synthetic fallback dataset
   - subsampling for persistent homology
   - lightweight autoencoders
   - small Mapper cover and modest graph sizes

2. Keep the number of outputs editorially useful:
   Section 5.1
       - state point cloud
       - persistence diagram
       - topological barcode
       - Betti curves
       - one summary table

   Section 5.2
       - pre-training reference manifold
       - topology-regularized latent manifold
       - topological evolution figure
       - before/after persistence image comparison
       - one model summary table
       - one checkpoint history table

   Section 5.3
       - topological knowledge map (KeplerMapper)
       - learning graph
       - one graph summary table
       - one recommendation alert table

3. Work with real data when available and otherwise produce a compact,
   reproducible synthetic educational dataset with meaningful multiscale
   structure.

Expected real-data files (all CSV, optional except interactions.csv)
--------------------------------------------------------------------
input_dir/
    interactions.csv
        Required columns after harmonization:
            learner_id, timestamp, concept_id, resource_id,
            score, correctness, difficulty, hints_used, dwell_time

        Accepted aliases include:
            student_id, user_id -> learner_id
            event_time, time -> timestamp
            skill_id, concept -> concept_id
            item_id, problem_id -> resource_id
            is_correct, correct -> correctness
            response_time, duration -> dwell_time

    resource_metadata.csv
        Optional columns:
            resource_id, concept_id, modality, estimated_duration, difficulty

    concept_edges.csv
        Optional columns:
            source, target, weight

Usage
-----
python chapter5_topology_pipeline.py \
    --input_dir ./data \
    --output_dir ./outputs \
    --window_size 8 \
    --window_stride 4 \
    --random_state 42

Recommended installation
------------------------
pip install numpy pandas matplotlib networkx scikit-learn scipy \
            gudhi ripser persim kmapper toponetx torch
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import gudhi as gd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ripser import ripser

try:
    from persim import plot_diagrams as persim_plot_diagrams
    PERSIM_AVAILABLE = True
except Exception:
    persim_plot_diagrams = None
    PERSIM_AVAILABLE = False

try:
    import kmapper as km
    KMAP_AVAILABLE = True
except Exception:
    km = None
    KMAP_AVAILABLE = False

try:
    import toponetx as tnx
    from toponetx.transform.graph_to_simplicial_complex import graph_to_clique_complex
    TOPONETX_AVAILABLE = True
except Exception:
    tnx = None
    graph_to_clique_complex = None
    TOPONETX_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    random_state: int = 42
    window_size: int = 8
    window_stride: int = 4
    min_window_events: int = 8
    max_points_ph: int = 220
    max_windows_for_latent: int = 480
    epochs: int = 36
    checkpoint_every: int = 6
    latent_dim: int = 3
    hidden_dim: int = 64
    k_neighbors: int = 8
    lambda_laplacian: float = 0.03
    lambda_stress: float = 0.02
    n_bins_betti: int = 72
    n_bins_persistence_image: int = 32
    fast_mode: bool = True
    use_synthetic_if_missing: bool = True

    @property
    def section_5_1_dir(self) -> Path:
        return self.output_dir / "5_1_coherence_stability_personalization"

    @property
    def section_5_2_dir(self) -> Path:
        return self.output_dir / "5_2_topological_regularization_latent_space"

    @property
    def section_5_3_dir(self) -> Path:
        return self.output_dir / "5_3_topological_recommendation_maps"


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def minmax_scale_array(values: Sequence[float], min_size: float, max_size: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if np.allclose(arr.max(), arr.min()):
        return np.full_like(arr, (min_size + max_size) / 2.0, dtype=float)
    scaled = (arr - arr.min()) / (arr.max() - arr.min())
    return min_size + scaled * (max_size - min_size)


def safe_entropy(values: Sequence[Any]) -> float:
    if len(values) == 0:
        return 0.0
    counts = pd.Series(list(values)).value_counts(normalize=True)
    return float(-(counts * np.log2(counts + 1e-12)).sum())


def parse_timestamp_column(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.datetime64):
        return series
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().all():
        return pd.to_datetime(pd.RangeIndex(start=0, stop=len(series), step=1), unit="s", origin="unix")
    return parsed


def adaptive_edge_length(points: np.ndarray, quantile: float = 0.35) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 1.0
    dists = pairwise_distances(points)
    tri = dists[np.triu_indices_from(dists, k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        return 1.0
    value = float(np.quantile(tri, quantile))
    return max(value, 1e-3)


def sample_indices_balanced(labels: Sequence[Any], max_items: int, random_state: int) -> np.ndarray:
    labels = np.asarray(labels)
    n = len(labels)
    if n <= max_items:
        return np.arange(n, dtype=int)

    rng = np.random.default_rng(random_state)
    unique_labels = pd.Series(labels).fillna("unknown").astype(str).unique().tolist()
    per_group = max(1, max_items // max(1, len(unique_labels)))
    selected: List[int] = []

    for label in unique_labels:
        idx = np.where(pd.Series(labels).fillna("unknown").astype(str).values == label)[0]
        take = min(per_group, len(idx))
        if take > 0:
            selected.extend(rng.choice(idx, size=take, replace=False).tolist())

    selected = sorted(set(selected))
    remaining = max_items - len(selected)
    if remaining > 0:
        others = np.array([i for i in range(n) if i not in selected], dtype=int)
        if len(others) > 0:
            extra = rng.choice(others, size=min(remaining, len(others)), replace=False).tolist()
            selected.extend(extra)

    return np.array(sorted(selected[:max_items]), dtype=int)


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    arr = np.asarray(diagram, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    arr = arr.reshape(-1, 2)
    arr = arr[np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])]
    return arr


def persistence_values(diagram: np.ndarray) -> np.ndarray:
    dgm = finite_diagram(diagram)
    if dgm.size == 0:
        return np.empty(0, dtype=float)
    pers = dgm[:, 1] - dgm[:, 0]
    pers = pers[np.isfinite(pers) & (pers >= 0)]
    return pers


def persistence_summary_from_dgms(dgms: Sequence[np.ndarray]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for dim, dgm in enumerate(dgms):
        pers = persistence_values(dgm)
        stats[f"h{dim}_features"] = float(len(pers))
        stats[f"h{dim}_mean_persistence"] = float(pers.mean()) if pers.size else 0.0
        stats[f"h{dim}_max_persistence"] = float(pers.max()) if pers.size else 0.0
        stats[f"h{dim}_sum_persistence"] = float(pers.sum()) if pers.size else 0.0
    return stats


def safe_bottleneck_distance(diag_a: np.ndarray, diag_b: np.ndarray) -> float:
    a = finite_diagram(diag_a)
    b = finite_diagram(diag_b)
    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0 or b.size == 0:
        return float("nan")
    try:
        return float(gd.bottleneck_distance(a, b))
    except Exception:
        return float("nan")


def normalize_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    scaler = MinMaxScaler()
    out = df.copy()
    if len(columns) == 0:
        return out
    out[list(columns)] = scaler.fit_transform(out[list(columns)])
    return out


def convert_alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "student_id": "learner_id",
        "user_id": "learner_id",
        "learner": "learner_id",
        "event_time": "timestamp",
        "time": "timestamp",
        "datetime": "timestamp",
        "skill_id": "concept_id",
        "concept": "concept_id",
        "topic_id": "concept_id",
        "item_id": "resource_id",
        "problem_id": "resource_id",
        "content_id": "resource_id",
        "is_correct": "correctness",
        "correct": "correctness",
        "response_time": "dwell_time",
        "duration": "dwell_time",
        "time_spent": "dwell_time",
    }
    rename_map = {col: aliases[col] for col in df.columns if col in aliases}
    return df.rename(columns=rename_map)


# ---------------------------------------------------------------------
# Lightweight replacements for giotto-tda components
# ---------------------------------------------------------------------

def _diagram_array(diagram: Optional[np.ndarray]) -> np.ndarray:
    if diagram is None:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(diagram, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    return arr.reshape(-1, 2)


def _finite_diagram_local(diagram: Optional[np.ndarray]) -> np.ndarray:
    arr = _diagram_array(diagram)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])]


def _as_diagram_batch(X: Any) -> List[List[np.ndarray]]:
    if isinstance(X, (list, tuple)):
        if len(X) == 0:
            return []

        first = X[0]
        if isinstance(first, (list, tuple)):
            batch: List[List[np.ndarray]] = []
            for sample in X:
                batch.append([_diagram_array(diag) for diag in sample])
            return batch

        if isinstance(first, np.ndarray) and first.ndim == 2 and first.shape[-1] == 2:
            return [[_diagram_array(diag) for diag in X]]

    if isinstance(X, np.ndarray) and X.dtype == object:
        batch: List[List[np.ndarray]] = []
        for sample in X:
            if isinstance(sample, (list, tuple)):
                batch.append([_diagram_array(diag) for diag in sample])
            elif isinstance(sample, np.ndarray) and sample.ndim == 2 and sample.shape[-1] == 2:
                batch.append([_diagram_array(sample)])
            else:
                raise TypeError("Unsupported object-array entry in diagram batch.")
        return batch

    raise TypeError("Unsupported input format for diagram batch.")


def _coerce_point_cloud_batch(X: Any) -> List[np.ndarray]:
    if isinstance(X, np.ndarray):
        if X.ndim == 2:
            return [np.asarray(X, dtype=float)]
        if X.ndim == 3:
            return [np.asarray(sample, dtype=float) for sample in X]
        raise ValueError(f"Expected a 2D or 3D array for point clouds, got shape {X.shape}.")
    if isinstance(X, (list, tuple)):
        if len(X) == 0:
            return []
        return [np.asarray(sample, dtype=float) for sample in X]
    raise TypeError("Unsupported input format for point-cloud batch.")


def _collect_finite_deaths(diagrams: Sequence[np.ndarray]) -> np.ndarray:
    vals: List[np.ndarray] = []
    for diag in diagrams:
        arr = _diagram_array(diag)
        if arr.size == 0:
            continue
        finite = arr[np.isfinite(arr[:, 1]), 1]
        if finite.size:
            vals.append(finite)
    if not vals:
        return np.empty(0, dtype=float)
    return np.concatenate(vals)


def _collect_finite_births(diagrams: Sequence[np.ndarray]) -> np.ndarray:
    vals: List[np.ndarray] = []
    for diag in diagrams:
        arr = _diagram_array(diag)
        if arr.size == 0:
            continue
        finite = arr[np.isfinite(arr[:, 0]), 0]
        if finite.size:
            vals.append(finite)
    if not vals:
        return np.empty(0, dtype=float)
    return np.concatenate(vals)


def _diagram_upper_for_betti(diagrams: Sequence[np.ndarray], fallback: float = 1.0) -> float:
    finite_deaths = _collect_finite_deaths(diagrams)
    finite_births = _collect_finite_births(diagrams)
    if finite_deaths.size > 0:
        upper = float(finite_deaths.max())
        if finite_births.size > 0:
            upper = max(upper, float(finite_births.max()))
        return max(upper, 1e-6)
    if finite_births.size > 0:
        return max(float(finite_births.max()) + 1.0, 1e-6)
    return max(fallback, 1e-6)


def _prepare_intervals_for_betti(diagram: np.ndarray, upper: float) -> np.ndarray:
    arr = _diagram_array(diagram)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    births = np.where(np.isfinite(arr[:, 0]), arr[:, 0], 0.0)
    deaths = np.where(np.isfinite(arr[:, 1]), arr[:, 1], upper)
    deaths = np.maximum(deaths, births)
    return np.column_stack([births, deaths])


def _betti_curve_from_intervals(intervals: np.ndarray, xs: np.ndarray) -> np.ndarray:
    if intervals.size == 0:
        return np.zeros_like(xs, dtype=float)
    births = intervals[:, 0][:, None]
    deaths = intervals[:, 1][:, None]
    alive = (births <= xs[None, :]) & (xs[None, :] < deaths)
    return alive.sum(axis=0).astype(float)


def _birth_persistence_points(diagram: np.ndarray) -> np.ndarray:
    arr = _finite_diagram_local(diagram)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    pers = arr[:, 1] - arr[:, 0]
    mask = np.isfinite(pers) & (pers > 0)
    if not np.any(mask):
        return np.empty((0, 2), dtype=float)
    return np.column_stack([arr[mask, 0], pers[mask]])


def _persistence_landscape_on_grid(diagram: np.ndarray, xs: np.ndarray, n_layers: int = 5) -> np.ndarray:
    arr = _finite_diagram_local(diagram)
    if arr.size == 0:
        return np.zeros((n_layers, len(xs)), dtype=float)

    births = arr[:, 0][:, None]
    deaths = arr[:, 1][:, None]
    tents = np.minimum(xs[None, :] - births, deaths - xs[None, :])
    tents = np.maximum(tents, 0.0)
    if tents.size == 0:
        return np.zeros((n_layers, len(xs)), dtype=float)

    sorted_vals = np.sort(tents, axis=0)[::-1]
    if sorted_vals.shape[0] < n_layers:
        pad = np.zeros((n_layers - sorted_vals.shape[0], sorted_vals.shape[1]), dtype=float)
        sorted_vals = np.vstack([sorted_vals, pad])
    return sorted_vals[:n_layers, :]


def plot_diagrams(diagrams: Sequence[np.ndarray], show: bool = False, ax: Optional[plt.Axes] = None, title: Optional[str] = None):
    if persim_plot_diagrams is not None:
        return persim_plot_diagrams(diagrams, show=show, ax=ax, title=title)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
    else:
        fig = ax.figure

    finite_vals: List[np.ndarray] = []
    for diag in diagrams:
        arr = _diagram_array(diag)
        if arr.size == 0:
            continue
        finite = arr[np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])]
        if finite.size:
            finite_vals.append(finite)

    if finite_vals:
        all_vals = np.vstack(finite_vals)
        lo = float(np.min(all_vals[:, 0]))
        hi = float(np.max(all_vals[:, 1]))
    else:
        lo, hi = 0.0, 1.0

    hi = max(hi, lo + 1e-3)
    colors = plt.cm.get_cmap("tab10", max(1, len(diagrams)))

    for dim, diag in enumerate(diagrams):
        arr = _diagram_array(diag)
        if arr.size == 0:
            continue
        births = np.where(np.isfinite(arr[:, 0]), arr[:, 0], lo)
        deaths = np.where(np.isfinite(arr[:, 1]), arr[:, 1], hi * 1.05)
        ax.scatter(
            births,
            deaths,
            s=26,
            alpha=0.8,
            color=colors(dim),
            edgecolors="white",
            linewidths=0.35,
            label=f"H{dim}",
        )

    diagonal = np.linspace(lo, hi * 1.05, 200)
    ax.plot(diagonal, diagonal, linestyle="--", linewidth=1.0, color="gray", alpha=0.7)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    if title:
        ax.set_title(title)
    if len(diagrams) > 1:
        ax.legend(frameon=False, fontsize=8)
    if show:
        plt.show()
    return fig


class VietorisRipsPersistence:
    def __init__(
        self,
        metric: str = "euclidean",
        homology_dimensions: Sequence[int] = (0, 1),
        collapse_edges: bool = True,
        max_edge_length: Optional[float] = None,
        n_jobs: int = 1,
    ):
        self.metric = metric
        self.homology_dimensions = tuple(sorted(set(int(dim) for dim in homology_dimensions)))
        self.collapse_edges = collapse_edges
        self.max_edge_length = max_edge_length
        self.n_jobs = n_jobs
        self.homology_dimensions_ = self.homology_dimensions

    def fit(self, X: Any, y: Optional[Any] = None) -> "VietorisRipsPersistence":
        return self

    def transform(self, X: Any) -> List[List[np.ndarray]]:
        point_clouds = _coerce_point_cloud_batch(X)
        batch: List[List[np.ndarray]] = []
        maxdim = max(self.homology_dimensions) if self.homology_dimensions else 0
        for points in point_clouds:
            res = ripser(
                np.asarray(points, dtype=float),
                maxdim=maxdim,
                thresh=self.max_edge_length if self.max_edge_length is not None else np.inf,
                metric=self.metric,
            )
            dgms = res["dgms"]
            selected: List[np.ndarray] = []
            for dim in self.homology_dimensions:
                if dim < len(dgms):
                    selected.append(_diagram_array(dgms[dim]))
                else:
                    selected.append(np.empty((0, 2), dtype=float))
            batch.append(selected)
        return batch

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> List[List[np.ndarray]]:
        return self.fit(X, y).transform(X)


class BettiCurve:
    def __init__(self, n_bins: int = 100):
        self.n_bins = int(n_bins)
        self.homology_dimensions_: Tuple[int, ...] = tuple()
        self.samplings_: Dict[int, np.ndarray] = {}
        self._uppers: Dict[int, float] = {}

    def fit(self, X: Any, y: Optional[Any] = None) -> "BettiCurve":
        batch = _as_diagram_batch(X)
        if not batch:
            self.homology_dimensions_ = tuple()
            self.samplings_ = {}
            self._uppers = {}
            return self

        n_dims = max((len(sample) for sample in batch), default=0)
        self.homology_dimensions_ = tuple(range(n_dims))
        self.samplings_ = {}
        self._uppers = {}

        for dim in self.homology_dimensions_:
            diagrams_dim = [sample[dim] if dim < len(sample) else np.empty((0, 2), dtype=float) for sample in batch]
            upper = _diagram_upper_for_betti(diagrams_dim, fallback=1.0)
            births = _collect_finite_births(diagrams_dim)
            lower = float(births.min()) if births.size else 0.0
            if not np.isfinite(lower):
                lower = 0.0
            if upper <= lower:
                upper = lower + 1.0
            self.samplings_[dim] = np.linspace(lower, upper, self.n_bins)
            self._uppers[dim] = upper
        return self

    def transform(self, X: Any) -> np.ndarray:
        batch = _as_diagram_batch(X)
        if not batch:
            return np.empty((0, 0, self.n_bins), dtype=float)
        if not self.homology_dimensions_:
            self.fit(batch)

        curves = np.zeros((len(batch), len(self.homology_dimensions_), self.n_bins), dtype=float)
        for i, sample in enumerate(batch):
            for j, dim in enumerate(self.homology_dimensions_):
                diag = sample[dim] if dim < len(sample) else np.empty((0, 2), dtype=float)
                xs = self.samplings_[dim]
                intervals = _prepare_intervals_for_betti(diag, upper=self._uppers[dim])
                curves[i, j, :] = _betti_curve_from_intervals(intervals, xs)
        return curves

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class PersistenceImage:
    def __init__(self, sigma: float = 0.08, n_bins: int = 32):
        self.sigma = float(sigma)
        self.n_bins = int(n_bins)
        self.homology_dimensions_: Tuple[int, ...] = tuple()
        self.birth_grid_: Dict[int, np.ndarray] = {}
        self.persistence_grid_: Dict[int, np.ndarray] = {}

    def fit(self, X: Any, y: Optional[Any] = None) -> "PersistenceImage":
        batch = _as_diagram_batch(X)
        if not batch:
            self.homology_dimensions_ = tuple()
            self.birth_grid_ = {}
            self.persistence_grid_ = {}
            return self

        n_dims = max((len(sample) for sample in batch), default=0)
        self.homology_dimensions_ = tuple(range(n_dims))
        self.birth_grid_ = {}
        self.persistence_grid_ = {}

        for dim in self.homology_dimensions_:
            pts_list = [_birth_persistence_points(sample[dim]) if dim < len(sample) else np.empty((0, 2), dtype=float) for sample in batch]
            pts = np.vstack([p for p in pts_list if p.size]) if any(p.size for p in pts_list) else np.empty((0, 2), dtype=float)

            if pts.size == 0:
                bmin, bmax = 0.0, 1.0
                pmin, pmax = 0.0, 1.0
            else:
                bmin, bmax = float(pts[:, 0].min()), float(pts[:, 0].max())
                pmin, pmax = 0.0, float(pts[:, 1].max())
                if bmax <= bmin:
                    bmax = bmin + 1.0
                if pmax <= pmin:
                    pmax = pmin + 1.0

            pad_b = 0.05 * max(bmax - bmin, 1.0)
            pad_p = 0.05 * max(pmax - pmin, 1.0)
            self.birth_grid_[dim] = np.linspace(bmin - pad_b, bmax + pad_b, self.n_bins)
            self.persistence_grid_[dim] = np.linspace(max(0.0, pmin), pmax + pad_p, self.n_bins)
        return self

    def transform(self, X: Any) -> np.ndarray:
        batch = _as_diagram_batch(X)
        if not batch:
            return np.empty((0, 0, self.n_bins, self.n_bins), dtype=float)
        if not self.homology_dimensions_:
            self.fit(batch)

        images = np.zeros((len(batch), len(self.homology_dimensions_), self.n_bins, self.n_bins), dtype=float)
        two_sigma2 = 2.0 * max(self.sigma ** 2, 1e-12)

        for i, sample in enumerate(batch):
            for j, dim in enumerate(self.homology_dimensions_):
                pts = _birth_persistence_points(sample[dim]) if dim < len(sample) else np.empty((0, 2), dtype=float)
                if pts.size == 0:
                    continue
                xs = self.birth_grid_[dim]
                ys = self.persistence_grid_[dim]
                XX, YY = np.meshgrid(xs, ys)
                image = np.zeros_like(XX, dtype=float)
                for birth, persistence in pts:
                    weight = float(max(persistence, 0.0))
                    image += weight * np.exp(-((XX - birth) ** 2 + (YY - persistence) ** 2) / two_sigma2)
                images[i, j, :, :] = image
        return images

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class PairwiseDistance:
    def __init__(self, metric: str = "landscape", order: float = 2.0, n_bins: int = 128, n_layers: int = 5):
        self.metric = str(metric).lower()
        self.order = float(order)
        self.n_bins = int(n_bins)
        self.n_layers = int(n_layers)

    def fit(self, X: Any, y: Optional[Any] = None) -> "PairwiseDistance":
        return self

    def _betti_distance(self, sample_a: List[np.ndarray], sample_b: List[np.ndarray]) -> float:
        n_dims = max(len(sample_a), len(sample_b))
        if n_dims == 0:
            return 0.0

        total = 0.0
        for dim in range(n_dims):
            diag_a = sample_a[dim] if dim < len(sample_a) else np.empty((0, 2), dtype=float)
            diag_b = sample_b[dim] if dim < len(sample_b) else np.empty((0, 2), dtype=float)
            upper = _diagram_upper_for_betti([diag_a, diag_b], fallback=1.0)
            births = _collect_finite_births([diag_a, diag_b])
            lower = float(births.min()) if births.size else 0.0
            if upper <= lower:
                upper = lower + 1.0
            xs = np.linspace(lower, upper, self.n_bins)
            curve_a = _betti_curve_from_intervals(_prepare_intervals_for_betti(diag_a, upper), xs)
            curve_b = _betti_curve_from_intervals(_prepare_intervals_for_betti(diag_b, upper), xs)
            total += float(np.trapz(np.abs(curve_a - curve_b) ** self.order, xs))
        return float(total ** (1.0 / self.order))

    def _landscape_distance(self, sample_a: List[np.ndarray], sample_b: List[np.ndarray]) -> float:
        n_dims = max(len(sample_a), len(sample_b))
        if n_dims == 0:
            return 0.0

        total = 0.0
        for dim in range(n_dims):
            diag_a = sample_a[dim] if dim < len(sample_a) else np.empty((0, 2), dtype=float)
            diag_b = sample_b[dim] if dim < len(sample_b) else np.empty((0, 2), dtype=float)

            finite_a = _finite_diagram_local(diag_a)
            finite_b = _finite_diagram_local(diag_b)
            if finite_a.size == 0 and finite_b.size == 0:
                continue

            vals = []
            if finite_a.size:
                vals.append(finite_a[:, 0])
                vals.append(finite_a[:, 1])
            if finite_b.size:
                vals.append(finite_b[:, 0])
                vals.append(finite_b[:, 1])
            all_vals = np.concatenate(vals)
            lower = float(all_vals.min())
            upper = float(all_vals.max())
            if upper <= lower:
                upper = lower + 1.0
            xs = np.linspace(lower, upper, self.n_bins)

            landscape_a = _persistence_landscape_on_grid(diag_a, xs, n_layers=self.n_layers)
            landscape_b = _persistence_landscape_on_grid(diag_b, xs, n_layers=self.n_layers)
            total += float(np.trapz(np.sum(np.abs(landscape_a - landscape_b) ** self.order, axis=0), xs))
        return float(total ** (1.0 / self.order))

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        batch = _as_diagram_batch(X)
        n = len(batch)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                if self.metric == "betti":
                    d = self._betti_distance(batch[i], batch[j])
                elif self.metric == "landscape":
                    d = self._landscape_distance(batch[i], batch[j])
                else:
                    raise ValueError(f"Unsupported metric: {self.metric}")
                D[i, j] = d
                D[j, i] = d
        return D



# ---------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------

def generate_synthetic_data(random_state: int = 42) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(random_state)

    n_learners = 96
    n_concepts = 12
    weeks = 16
    interactions_per_week = 6

    concepts = [f"C{idx:02d}" for idx in range(1, n_concepts + 1)]
    modalities = ["video", "reading", "quiz", "simulation"]
    concept_difficulty = {
        concept: 0.9 + 0.1 * idx + 0.12 * math.sin(idx / 2)
        for idx, concept in enumerate(concepts, start=1)
    }

    resources_rows = []
    for concept in concepts:
        for rid in range(1, 5):
            resources_rows.append(
                {
                    "resource_id": f"R_{concept}_{rid}",
                    "concept_id": concept,
                    "modality": modalities[(rid - 1) % len(modalities)],
                    "estimated_duration": int(rng.integers(5, 18)),
                    "difficulty": concept_difficulty[concept] + float(rng.normal(0, 0.05)),
                }
            )
    resources = pd.DataFrame(resources_rows)

    concept_edges_rows = []
    for i in range(len(concepts) - 1):
        concept_edges_rows.append(
            {"source": concepts[i], "target": concepts[i + 1], "weight": 1.0}
        )
    concept_edges_rows.extend(
        [
            {"source": "C03", "target": "C05", "weight": 0.7},
            {"source": "C04", "target": "C07", "weight": 0.6},
            {"source": "C06", "target": "C08", "weight": 0.7},
            {"source": "C08", "target": "C10", "weight": 0.7},
            {"source": "C10", "target": "C12", "weight": 0.8},
            {"source": "C05", "target": "C03", "weight": 0.35},
            {"source": "C09", "target": "C06", "weight": 0.30},
        ]
    )
    concept_edges = pd.DataFrame(concept_edges_rows)

    profile_rows = []
    interactions_rows = []

    base_date = pd.Timestamp("2026-01-05")

    groups = ["stable", "exploratory", "struggling"]
    group_probs = [0.42, 0.28, 0.30]

    for learner_idx in range(1, n_learners + 1):
        learner_id = f"L{learner_idx:03d}"
        trajectory_group = str(rng.choice(groups, p=group_probs))
        ability = {
            "stable": float(rng.normal(0.85, 0.08)),
            "exploratory": float(rng.normal(0.72, 0.10)),
            "struggling": float(rng.normal(0.58, 0.09)),
        }[trajectory_group]
        discipline = {
            "stable": float(rng.normal(0.80, 0.08)),
            "exploratory": float(rng.normal(0.65, 0.10)),
            "struggling": float(rng.normal(0.52, 0.09)),
        }[trajectory_group]
        review_bias = {
            "stable": 0.15,
            "exploratory": 0.25,
            "struggling": 0.45,
        }[trajectory_group]
        risk_group = {
            "stable": "low_risk",
            "exploratory": "medium_risk",
            "struggling": "high_risk",
        }[trajectory_group]

        profile_rows.append(
            {
                "learner_id": learner_id,
                "trajectory_group": trajectory_group,
                "risk_group": risk_group,
                "ability": ability,
                "discipline": discipline,
            }
        )

        current_concept_idx = 0
        mastered = {concept: 0.0 for concept in concepts}
        timestamp_cursor = base_date + pd.Timedelta(days=int(rng.integers(0, 3)))

        for week in range(1, weeks + 1):
            for attempt in range(interactions_per_week):
                if trajectory_group == "stable":
                    if rng.random() < review_bias and current_concept_idx > 1:
                        chosen_idx = max(0, current_concept_idx - int(rng.integers(1, 3)))
                    else:
                        chosen_idx = min(
                            len(concepts) - 1,
                            current_concept_idx + int(rng.integers(0, 2)),
                        )
                elif trajectory_group == "exploratory":
                    jump = int(rng.choice([-2, -1, 0, 1, 2], p=[0.10, 0.18, 0.26, 0.28, 0.18]))
                    chosen_idx = int(np.clip(current_concept_idx + jump, 0, len(concepts) - 1))
                else:
                    if rng.random() < 0.55 and current_concept_idx > 0:
                        chosen_idx = max(0, current_concept_idx - 1)
                    else:
                        chosen_idx = min(
                            len(concepts) - 1,
                            current_concept_idx + int(rng.integers(0, 2)),
                        )

                concept_id = concepts[chosen_idx]
                current_concept_idx = chosen_idx if rng.random() > 0.35 else current_concept_idx

                concept_base_difficulty = concept_difficulty[concept_id]
                difficulty = max(0.4, concept_base_difficulty + float(rng.normal(0, 0.08)))

                support_mode = rng.choice(
                    ["practice", "review", "hint", "challenge"],
                    p={
                        "stable": [0.46, 0.22, 0.12, 0.20],
                        "exploratory": [0.34, 0.18, 0.18, 0.30],
                        "struggling": [0.28, 0.26, 0.30, 0.16],
                    }[trajectory_group],
                )

                mastery_bonus = 0.22 * mastered[concept_id]
                progression_bonus = 0.02 * week
                support_bonus = {
                    "practice": 0.00,
                    "review": 0.03,
                    "hint": 0.06,
                    "challenge": -0.03,
                }[support_mode]

                latent_success = (
                    2.2 * ability
                    + 0.5 * discipline
                    + mastery_bonus
                    + progression_bonus
                    + support_bonus
                    - 1.8 * difficulty
                    + float(rng.normal(0, 0.18))
                )
                prob_correct = 1.0 / (1.0 + np.exp(-latent_success))
                correctness = int(rng.random() < prob_correct)

                hints_used = int(
                    max(
                        0,
                        rng.poisson(
                            lam=max(
                                0.2,
                                1.0 + 1.3 * (1 - prob_correct) + (0.4 if trajectory_group == "struggling" else 0.0),
                            )
                        )
                    )
                )
                dwell_time = float(
                    np.clip(
                        rng.normal(
                            loc=7.5 + 4.5 * difficulty + 1.4 * hints_used - 1.0 * correctness,
                            scale=2.0,
                        ),
                        1.0,
                        30.0,
                    )
                )
                score = float(
                    np.clip(
                        38
                        + 48 * prob_correct
                        + 8 * correctness
                        - 2.5 * hints_used
                        - 2.0 * difficulty
                        + rng.normal(0, 5),
                        0,
                        100,
                    )
                )
                if correctness:
                    mastered[concept_id] = min(1.0, mastered[concept_id] + 0.11 + 0.02 * discipline)
                else:
                    mastered[concept_id] = max(0.0, mastered[concept_id] - 0.03)

                concept_resources = resources.loc[resources["concept_id"] == concept_id, "resource_id"].tolist()
                resource_id = str(rng.choice(concept_resources))

                timestamp_cursor = timestamp_cursor + pd.Timedelta(
                    days=int(rng.integers(0, 2)),
                    hours=int(rng.integers(1, 8)),
                    minutes=int(rng.integers(0, 55)),
                )

                interactions_rows.append(
                    {
                        "learner_id": learner_id,
                        "timestamp": timestamp_cursor,
                        "week": week,
                        "concept_id": concept_id,
                        "resource_id": resource_id,
                        "score": score,
                        "correctness": correctness,
                        "difficulty": difficulty,
                        "hints_used": hints_used,
                        "dwell_time": dwell_time,
                        "tutor_action": support_mode,
                        "trajectory_group": trajectory_group,
                        "risk_group": risk_group,
                    }
                )

            if trajectory_group == "stable" and week % 2 == 0:
                current_concept_idx = min(len(concepts) - 1, current_concept_idx + 1)
            elif trajectory_group == "exploratory":
                current_concept_idx = int(np.clip(current_concept_idx + rng.integers(-1, 2), 0, len(concepts) - 1))
            else:
                if rng.random() > 0.35:
                    current_concept_idx = min(len(concepts) - 1, current_concept_idx + 1)

    interactions = pd.DataFrame(interactions_rows).sort_values(["learner_id", "timestamp"]).reset_index(drop=True)
    learner_profiles = pd.DataFrame(profile_rows)

    return {
        "interactions": interactions,
        "resources": resources,
        "concept_edges": concept_edges,
        "learner_profiles": learner_profiles,
    }


# ---------------------------------------------------------------------
# Real data loading and harmonization
# ---------------------------------------------------------------------

def load_real_data(input_dir: Path) -> Dict[str, pd.DataFrame]:
    interactions_path = input_dir / "interactions.csv"
    if not interactions_path.exists():
        raise FileNotFoundError(f"Missing required file: {interactions_path}")

    interactions = pd.read_csv(interactions_path)
    interactions = convert_alias_columns(interactions)

    if "learner_id" not in interactions.columns:
        raise ValueError("interactions.csv must contain learner_id (or an accepted alias).")

    if "timestamp" not in interactions.columns:
        interactions["timestamp"] = np.arange(len(interactions), dtype=int)

    if "concept_id" not in interactions.columns and "resource_id" in interactions.columns:
        interactions["concept_id"] = interactions["resource_id"].astype(str)
    elif "concept_id" not in interactions.columns:
        raise ValueError("interactions.csv must contain concept_id (or an accepted alias).")

    if "resource_id" not in interactions.columns:
        interactions["resource_id"] = interactions["concept_id"].astype(str) + "_resource"

    if "correctness" not in interactions.columns:
        if "score" in interactions.columns:
            interactions["correctness"] = (pd.to_numeric(interactions["score"], errors="coerce").fillna(0) >= 60).astype(int)
        else:
            interactions["correctness"] = 0

    if "score" not in interactions.columns:
        interactions["score"] = interactions["correctness"].astype(float) * 100.0

    if "difficulty" not in interactions.columns:
        interactions["difficulty"] = 1.0

    if "hints_used" not in interactions.columns:
        interactions["hints_used"] = 0

    if "dwell_time" not in interactions.columns:
        interactions["dwell_time"] = 5.0

    if "tutor_action" not in interactions.columns:
        interactions["tutor_action"] = np.where(interactions["hints_used"] > 0, "hint", "practice")

    interactions["timestamp"] = parse_timestamp_column(interactions["timestamp"])
    interactions["score"] = pd.to_numeric(interactions["score"], errors="coerce").fillna(0).astype(float)
    interactions["correctness"] = pd.to_numeric(interactions["correctness"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    interactions["difficulty"] = pd.to_numeric(interactions["difficulty"], errors="coerce").fillna(1.0).astype(float)
    interactions["hints_used"] = pd.to_numeric(interactions["hints_used"], errors="coerce").fillna(0).astype(float)
    interactions["dwell_time"] = pd.to_numeric(interactions["dwell_time"], errors="coerce").fillna(5.0).astype(float)
    interactions = interactions.sort_values(["learner_id", "timestamp"]).reset_index(drop=True)

    resources_path = input_dir / "resource_metadata.csv"
    if resources_path.exists():
        resources = pd.read_csv(resources_path)
        if "resource_id" not in resources.columns:
            raise ValueError("resource_metadata.csv must contain resource_id.")
        if "concept_id" not in resources.columns:
            resources["concept_id"] = resources["resource_id"].astype(str)
        if "modality" not in resources.columns:
            resources["modality"] = "resource"
        if "estimated_duration" not in resources.columns:
            resources["estimated_duration"] = 10
        if "difficulty" not in resources.columns:
            resources["difficulty"] = 1.0
    else:
        resources = (
            interactions[["resource_id", "concept_id"]]
            .drop_duplicates()
            .assign(modality="resource", estimated_duration=10, difficulty=1.0)
        )

    concept_edges_path = input_dir / "concept_edges.csv"
    if concept_edges_path.exists():
        concept_edges = pd.read_csv(concept_edges_path)
        if "source" not in concept_edges.columns or "target" not in concept_edges.columns:
            raise ValueError("concept_edges.csv must contain source and target columns.")
        if "weight" not in concept_edges.columns:
            concept_edges["weight"] = 1.0
    else:
        concept_edges = infer_concept_edges_from_interactions(interactions)

    learner_profiles = infer_learner_profiles(interactions)

    return {
        "interactions": interactions,
        "resources": resources,
        "concept_edges": concept_edges,
        "learner_profiles": learner_profiles,
    }


def infer_concept_edges_from_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    counts: Dict[Tuple[str, str], int] = {}
    for _, learner_df in interactions.sort_values(["learner_id", "timestamp"]).groupby("learner_id"):
        concepts = learner_df["concept_id"].astype(str).tolist()
        for a, b in zip(concepts[:-1], concepts[1:]):
            if a != b:
                counts[(a, b)] = counts.get((a, b), 0) + 1
    for (source, target), weight in counts.items():
        rows.append({"source": source, "target": target, "weight": float(weight)})
    if not rows:
        unique_concepts = interactions["concept_id"].astype(str).drop_duplicates().tolist()
        for a, b in zip(unique_concepts[:-1], unique_concepts[1:]):
            rows.append({"source": a, "target": b, "weight": 1.0})
    return pd.DataFrame(rows)


def infer_learner_profiles(interactions: pd.DataFrame) -> pd.DataFrame:
    agg = (
        interactions.groupby("learner_id")
        .agg(
            mean_score=("score", "mean"),
            mean_correctness=("correctness", "mean"),
            mean_hints=("hints_used", "mean"),
            mean_difficulty=("difficulty", "mean"),
        )
        .reset_index()
    )
    score_q1 = agg["mean_score"].quantile(0.33)
    score_q2 = agg["mean_score"].quantile(0.66)
    hints_q = agg["mean_hints"].quantile(0.66)

    def assign_group(row: pd.Series) -> str:
        if row["mean_score"] >= score_q2 and row["mean_hints"] < hints_q:
            return "stable"
        if row["mean_score"] <= score_q1:
            return "struggling"
        return "exploratory"

    agg["trajectory_group"] = agg.apply(assign_group, axis=1)
    agg["risk_group"] = np.where(agg["trajectory_group"] == "struggling", "high_risk",
                          np.where(agg["trajectory_group"] == "stable", "low_risk", "medium_risk"))
    agg["ability"] = normalize_columns(agg, ["mean_score"])["mean_score"]
    agg["discipline"] = 1.0 - normalize_columns(agg, ["mean_hints"])["mean_hints"]
    return agg[["learner_id", "trajectory_group", "risk_group", "ability", "discipline"]]


def load_or_generate_data(config: PipelineConfig) -> Dict[str, pd.DataFrame]:
    try:
        return load_real_data(config.input_dir)
    except Exception as exc:
        if not config.use_synthetic_if_missing:
            raise
        warnings.warn(
            f"Real data could not be loaded from {config.input_dir}. "
            f"Falling back to a compact synthetic dataset. Reason: {exc}"
        )
        return generate_synthetic_data(random_state=config.random_state)


# ---------------------------------------------------------------------
# Feature engineering for windows and concepts
# ---------------------------------------------------------------------

def build_learner_windows(
    interactions: pd.DataFrame,
    resources: pd.DataFrame,
    learner_profiles: pd.DataFrame,
    window_size: int,
    window_stride: int,
    min_window_events: int,
) -> pd.DataFrame:
    merged = interactions.copy()
    merged["timestamp"] = parse_timestamp_column(merged["timestamp"])
    if resources is not None and not resources.empty:
        resource_cols = [col for col in ["resource_id", "concept_id", "modality"] if col in resources.columns]
        merged = merged.merge(resources[resource_cols].drop_duplicates(), on=["resource_id", "concept_id"], how="left")
    if "modality" not in merged.columns:
        merged["modality"] = "resource"

    if learner_profiles is not None and not learner_profiles.empty:
        merged = merged.merge(
            learner_profiles[["learner_id", "trajectory_group", "risk_group"]].drop_duplicates(),
            on="learner_id",
            how="left",
            suffixes=("", "_profile"),
        )
        if "trajectory_group_profile" in merged.columns:
            merged["trajectory_group"] = merged["trajectory_group"].fillna(merged["trajectory_group_profile"])
            merged.drop(columns=["trajectory_group_profile"], inplace=True)
        if "risk_group_profile" in merged.columns:
            merged["risk_group"] = merged["risk_group"].fillna(merged["risk_group_profile"])
            merged.drop(columns=["risk_group_profile"], inplace=True)

    top_concepts = (
        merged["concept_id"].astype(str).value_counts().head(12).index.tolist()
    )
    top_modalities = merged["modality"].astype(str).value_counts().head(4).index.tolist()

    rows: List[Dict[str, Any]] = []
    for learner_id, df in merged.sort_values(["learner_id", "timestamp"]).groupby("learner_id"):
        df = df.reset_index(drop=True)
        if len(df) < min_window_events:
            continue
        total_windows = max(1, int(math.ceil((len(df) - window_size) / max(window_stride, 1))) + 1)
        window_counter = 0
        for start in range(0, max(1, len(df) - window_size + 1), max(1, window_stride)):
            window = df.iloc[start:start + window_size].copy()
            if len(window) < min_window_events:
                continue

            concepts = window["concept_id"].astype(str).tolist()
            modalities = window["modality"].astype(str).tolist()
            transitions = list(zip(concepts[:-1], concepts[1:]))

            row: Dict[str, Any] = {
                "learner_id": learner_id,
                "window_id": f"{learner_id}_W{window_counter:03d}",
                "window_index": window_counter,
                "window_position_norm": window_counter / max(total_windows - 1, 1),
                "n_events": int(len(window)),
                "trajectory_group": str(window["trajectory_group"].iloc[0]) if "trajectory_group" in window.columns else "unknown",
                "risk_group": str(window["risk_group"].iloc[0]) if "risk_group" in window.columns else "unknown",
                "mean_score": float(window["score"].mean()),
                "std_score": float(window["score"].std(ddof=0)) if len(window) > 1 else 0.0,
                "correctness_rate": float(window["correctness"].mean()),
                "mean_difficulty": float(window["difficulty"].mean()),
                "std_difficulty": float(window["difficulty"].std(ddof=0)) if len(window) > 1 else 0.0,
                "mean_hints": float(window["hints_used"].mean()),
                "sum_hints": float(window["hints_used"].sum()),
                "mean_dwell_time": float(window["dwell_time"].mean()),
                "std_dwell_time": float(window["dwell_time"].std(ddof=0)) if len(window) > 1 else 0.0,
                "unique_concepts": float(pd.Series(concepts).nunique()),
                "revisit_ratio": float(1.0 - pd.Series(concepts).nunique() / max(len(concepts), 1)),
                "concept_entropy": safe_entropy(concepts),
                "transition_diversity": float(pd.Series(transitions).nunique() / max(len(transitions), 1)) if transitions else 0.0,
                "modality_entropy": safe_entropy(modalities),
                "review_action_rate": float((window["tutor_action"].astype(str) == "review").mean()) if "tutor_action" in window.columns else 0.0,
                "hint_action_rate": float((window["tutor_action"].astype(str) == "hint").mean()) if "tutor_action" in window.columns else 0.0,
                "challenge_action_rate": float((window["tutor_action"].astype(str) == "challenge").mean()) if "tutor_action" in window.columns else 0.0,
                "week_mean": float(window["week"].mean()) if "week" in window.columns else float(window_counter),
            }

            concept_counts = pd.Series(concepts).value_counts(normalize=True)
            for concept in top_concepts:
                row[f"concept_freq_{concept}"] = float(concept_counts.get(concept, 0.0))

            modality_counts = pd.Series(modalities).value_counts(normalize=True)
            for modality in top_modalities:
                row[f"modality_freq_{modality}"] = float(modality_counts.get(modality, 0.0))

            rows.append(row)
            window_counter += 1

    return pd.DataFrame(rows)


def prepare_window_matrix(window_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    ignore_cols = {"learner_id", "window_id", "trajectory_group", "risk_group"}
    feature_cols = [c for c in window_df.columns if c not in ignore_cols]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(window_df[c])]
    work = window_df[feature_cols].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X = imputer.fit_transform(work)
    X = scaler.fit_transform(X)
    meta = window_df[["learner_id", "window_id", "trajectory_group", "risk_group"]].copy()
    return X, meta, feature_cols


def build_transition_graph(interactions: pd.DataFrame, concept_edges: pd.DataFrame) -> Tuple[nx.DiGraph, pd.DataFrame]:
    graph = nx.DiGraph()

    concept_summary = (
        interactions.groupby("concept_id")
        .agg(
            usage=("concept_id", "size"),
            mean_score=("score", "mean"),
            success_rate=("correctness", "mean"),
            mean_difficulty=("difficulty", "mean"),
            mean_hints=("hints_used", "mean"),
            mean_dwell_time=("dwell_time", "mean"),
        )
        .reset_index()
    )
    concept_summary["failure_rate"] = 1.0 - concept_summary["success_rate"]

    for _, row in concept_summary.iterrows():
        graph.add_node(
            str(row["concept_id"]),
            usage=float(row["usage"]),
            mean_score=float(row["mean_score"]),
            success_rate=float(row["success_rate"]),
            failure_rate=float(row["failure_rate"]),
            mean_difficulty=float(row["mean_difficulty"]),
            mean_hints=float(row["mean_hints"]),
            mean_dwell_time=float(row["mean_dwell_time"]),
        )

    transition_counts: Dict[Tuple[str, str], int] = {}
    for _, df in interactions.sort_values(["learner_id", "timestamp"]).groupby("learner_id"):
        concepts = df["concept_id"].astype(str).tolist()
        for a, b in zip(concepts[:-1], concepts[1:]):
            if a == b:
                continue
            transition_counts[(a, b)] = transition_counts.get((a, b), 0) + 1

    for (a, b), w in transition_counts.items():
        if graph.has_edge(a, b):
            graph[a][b]["weight"] += float(w)
            graph[a][b]["observed_transition_count"] += int(w)
        else:
            graph.add_edge(a, b, weight=float(w), observed_transition_count=int(w), source="interaction")

    if concept_edges is not None and not concept_edges.empty:
        for _, row in concept_edges.iterrows():
            source = str(row["source"])
            target = str(row["target"])
            w = float(row.get("weight", 1.0))
            if graph.has_edge(source, target):
                graph[source][target]["weight"] += w
                graph[source][target]["source"] = "interaction+prerequisite"
            else:
                graph.add_edge(source, target, weight=w, observed_transition_count=0, source="prerequisite")

    return graph, concept_summary


def build_concept_feature_table(graph: nx.DiGraph, concept_summary: pd.DataFrame) -> pd.DataFrame:
    undirected = graph.to_undirected()
    pagerank = nx.pagerank(graph, alpha=0.90, weight="weight") if graph.number_of_nodes() > 0 else {}
    betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True) if graph.number_of_nodes() > 0 else {}
    degree = dict(graph.degree(weight="weight"))
    indegree = dict(graph.in_degree(weight="weight"))
    outdegree = dict(graph.out_degree(weight="weight"))

    df = concept_summary.copy()
    df["pagerank"] = df["concept_id"].astype(str).map(pagerank).fillna(0.0)
    df["betweenness"] = df["concept_id"].astype(str).map(betweenness).fillna(0.0)
    df["degree_weighted"] = df["concept_id"].astype(str).map(degree).fillna(0.0)
    df["indegree_weighted"] = df["concept_id"].astype(str).map(indegree).fillna(0.0)
    df["outdegree_weighted"] = df["concept_id"].astype(str).map(outdegree).fillna(0.0)
    clustering = nx.clustering(undirected, weight="weight") if undirected.number_of_nodes() > 0 else {}
    df["local_clustering"] = df["concept_id"].astype(str).map(clustering).fillna(0.0)

    score_cols = ["failure_rate", "betweenness", "pagerank"]
    scaled = normalize_columns(df[["concept_id"] + score_cols].copy(), score_cols)
    df["priority_score"] = (
        0.45 * scaled["failure_rate"] + 0.35 * scaled["betweenness"] + 0.20 * scaled["pagerank"]
    )
    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------
# Section 5.1: persistent topology of learning windows
# ---------------------------------------------------------------------

def plot_state_point_cloud(points_2d: np.ndarray, labels: Sequence[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    labels = pd.Series(labels).fillna("unknown").astype(str)
    unique_labels = sorted(labels.unique().tolist())
    cmap = plt.cm.get_cmap("tab10", max(1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            points_2d[mask, 0],
            points_2d[mask, 1],
            s=32,
            alpha=0.78,
            label=label,
            color=cmap(idx),
            edgecolors="white",
            linewidths=0.4,
        )

    ax.set_title("Figure 5.1A. Learning-state point cloud")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)
    save_figure(fig, path)


def plot_persistence_diagram_figure(dgms: Sequence[np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    plot_diagrams(dgms, show=False, ax=ax, title="Figure 5.1B. Persistence diagram")
    ax.grid(alpha=0.25, linewidth=0.5)
    save_figure(fig, path)


def plot_barcode_figure(persistence: Sequence[Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    gd.plot_persistence_barcode(persistence=persistence, legend=True, axes=ax)
    ax.set_title("Figure 5.1C. Topological barcode")
    ax.grid(alpha=0.18, linewidth=0.4)
    save_figure(fig, path)


def fit_betti_curves(diagrams_batch: np.ndarray, n_bins: int) -> Tuple[np.ndarray, BettiCurve]:
    transformer = BettiCurve(n_bins=n_bins)
    curves = transformer.fit_transform(diagrams_batch)
    return curves, transformer


def plot_betti_curves(curves: np.ndarray, transformer: BettiCurve, path: Path) -> None:
    dims = list(transformer.homology_dimensions_)
    curves_sample = np.squeeze(curves[0])

    if curves_sample.ndim == 1:
        curves_sample = curves_sample[None, :]
    elif curves_sample.ndim == 2 and curves_sample.shape[0] != len(dims) and curves_sample.shape[1] == len(dims):
        curves_sample = curves_sample.T

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for idx, dim in enumerate(dims):
        xs = np.asarray(transformer.samplings_[dim], dtype=float)
        ys = np.asarray(curves_sample[idx], dtype=float)
        ax.plot(xs, ys, linewidth=2.0, label=f"H{dim}")

    ax.set_title("Figure 5.1D. Betti curves")
    ax.set_xlabel("Filtration value")
    ax.set_ylabel("Betti number")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    save_figure(fig, path)


def run_section_5_1(
    config: PipelineConfig,
    window_df: pd.DataFrame,
    X_windows: np.ndarray,
    meta_windows: pd.DataFrame,
) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(config.section_5_1_dir)
    manifest: List[Dict[str, Any]] = []

    labels = meta_windows["trajectory_group"].astype(str).fillna("unknown").values
    idx = sample_indices_balanced(labels, config.max_points_ph, config.random_state)
    X_sample = X_windows[idx]
    meta_sample = meta_windows.iloc[idx].reset_index(drop=True)

    pca2 = PCA(n_components=2, random_state=config.random_state)
    points_2d = pca2.fit_transform(X_sample)
    point_cloud_path = out_dir / "figure_5_1a_state_point_cloud.png"
    plot_state_point_cloud(points_2d, meta_sample["trajectory_group"].tolist(), point_cloud_path)
    manifest.append(
        {
            "section": "5.1",
            "type": "figure",
            "file": str(point_cloud_path),
            "purpose": "Base visual map of learning-state windows used for topology estimation.",
        }
    )

    ph_points = PCA(n_components=min(5, X_sample.shape[1]), random_state=config.random_state).fit_transform(X_sample)
    max_edge_length = adaptive_edge_length(ph_points, quantile=0.35)

    dgms = ripser(ph_points, maxdim=1, thresh=max_edge_length)["dgms"]
    diag_path = out_dir / "figure_5_1b_persistence_diagram.png"
    plot_persistence_diagram_figure(dgms, diag_path)
    manifest.append(
        {
            "section": "5.1",
            "type": "figure",
            "file": str(diag_path),
            "purpose": "Persistence diagram for multiscale coherence and cycle structure.",
        }
    )

    rips_complex = gd.RipsComplex(points=ph_points.tolist(), max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    persistence = simplex_tree.persistence()

    barcode_path = out_dir / "figure_5_1c_topological_barcode.png"
    plot_barcode_figure(persistence, barcode_path)
    manifest.append(
        {
            "section": "5.1",
            "type": "figure",
            "file": str(barcode_path),
            "purpose": "Barcode representation of stable connected components and loops.",
        }
    )

    vr = VietorisRipsPersistence(
        metric="euclidean",
        homology_dimensions=(0, 1),
        collapse_edges=True,
        max_edge_length=max_edge_length,
        n_jobs=1,
    )
    diagrams_batch = vr.fit_transform(ph_points[None, :, :])
    betti_curves, betti_transformer = fit_betti_curves(diagrams_batch, n_bins=config.n_bins_betti)

    betti_path = out_dir / "figure_5_1d_betti_curves.png"
    plot_betti_curves(betti_curves, betti_transformer, betti_path)
    manifest.append(
        {
            "section": "5.1",
            "type": "figure",
            "file": str(betti_path),
            "purpose": "Betti curves summarizing the persistence of H0 and H1 across scales.",
        }
    )

    rows: List[Dict[str, Any]] = []
    overall_stats = persistence_summary_from_dgms(dgms)
    overall_stats.update(
        {
            "group": "overall",
            "n_windows": int(len(ph_points)),
            "simplex_tree_vertices": int(simplex_tree.num_vertices()),
            "simplex_tree_simplices": int(simplex_tree.num_simplices()),
            "max_edge_length": float(max_edge_length),
        }
    )
    rows.append(overall_stats)

    for group_name, group_meta in meta_sample.groupby("trajectory_group"):
        group_idx = group_meta.index.to_numpy()
        if len(group_idx) < 15:
            continue
        group_points = ph_points[group_idx]
        group_thresh = adaptive_edge_length(group_points, quantile=0.35)
        group_dgms = ripser(group_points, maxdim=1, thresh=group_thresh)["dgms"]
        stats = persistence_summary_from_dgms(group_dgms)
        stats.update(
            {
                "group": str(group_name),
                "n_windows": int(len(group_points)),
                "simplex_tree_vertices": int(len(group_points)),
                "simplex_tree_simplices": float("nan"),
                "max_edge_length": float(group_thresh),
            }
        )
        rows.append(stats)

    table_df = pd.DataFrame(rows)
    table_path = out_dir / "table_5_1_topological_summary.csv"
    table_df.to_csv(table_path, index=False)
    manifest.append(
        {
            "section": "5.1",
            "type": "table",
            "file": str(table_path),
            "purpose": "Summary of H0/H1 persistence by trajectory group and overall cohort.",
        }
    )

    return manifest


# ---------------------------------------------------------------------
# Section 5.2: topology-regularized latent space
# ---------------------------------------------------------------------

class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 3):
        super().__init__()
        bottleneck = max(16, hidden_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


def build_knn_adjacency(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    n_neighbors = min(max(2, n_neighbors), max(2, len(X) - 1))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn_model.fit(X)
    _, indices = nn_model.kneighbors(X)
    A = np.zeros((len(X), len(X)), dtype=np.float32)
    for i in range(len(X)):
        for j in indices[i, 1:]:
            A[i, j] = 1.0
            A[j, i] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


def laplacian_penalty(z: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    diff = z.unsqueeze(1) - z.unsqueeze(0)
    sqdist = (diff ** 2).sum(dim=-1)
    weighted = sqdist * adjacency
    denom = adjacency.sum().clamp_min(1.0)
    return weighted.sum() / denom


def stress_penalty(z_ref: torch.Tensor, d_ref: torch.Tensor) -> torch.Tensor:
    d_latent = torch.cdist(z_ref, z_ref, p=2)
    return F.mse_loss(d_latent, d_ref)


def compute_trustworthiness(X: np.ndarray, Z: np.ndarray, n_neighbors: int = 10) -> float:
    if len(X) <= max(2, n_neighbors + 1):
        return float("nan")
    try:
        return float(trustworthiness(X, Z, n_neighbors=min(n_neighbors, len(X) - 1)))
    except Exception:
        return float("nan")


def compute_vr_diagrams_for_batch(batch_points: np.ndarray, max_edge_length: float) -> np.ndarray:
    vr = VietorisRipsPersistence(
        metric="euclidean",
        homology_dimensions=(0, 1),
        collapse_edges=True,
        max_edge_length=max_edge_length,
        n_jobs=1,
    )
    return vr.fit_transform(batch_points)


def summarize_latent_topology(points: np.ndarray, reference_dgms: Optional[Sequence[np.ndarray]] = None) -> Tuple[Dict[str, float], Sequence[np.ndarray]]:
    thresh = adaptive_edge_length(points, quantile=0.35)
    dgms = ripser(points, maxdim=1, thresh=thresh)["dgms"]
    stats = persistence_summary_from_dgms(dgms)
    stats["tda_threshold"] = float(thresh)
    if reference_dgms is not None:
        stats["h1_bottleneck_to_reference"] = safe_bottleneck_distance(reference_dgms[1], dgms[1])
        stats["h0_bottleneck_to_reference"] = safe_bottleneck_distance(reference_dgms[0], dgms[0])
    else:
        stats["h1_bottleneck_to_reference"] = float("nan")
        stats["h0_bottleneck_to_reference"] = float("nan")
    return stats, dgms


def train_autoencoder(
    X: np.ndarray,
    labels: Sequence[str],
    config: PipelineConfig,
    lambda_laplacian: float,
    lambda_stress: float,
    model_name: str,
) -> Tuple[AutoEncoder, pd.DataFrame]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for Section 5.2. Install torch before running the script.")

    device = torch.device("cpu")
    model = AutoEncoder(
        input_dim=X.shape[1],
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    adjacency = torch.tensor(build_knn_adjacency(X, config.k_neighbors), dtype=torch.float32, device=device)

    ref_size = min(120, len(X))
    ref_idx = sample_indices_balanced(labels, ref_size, config.random_state + 13)
    ref_idx_t = torch.tensor(ref_idx, dtype=torch.long, device=device)
    d_ref = torch.tensor(pairwise_distances(X[ref_idx]), dtype=torch.float32, device=device)

    tda_size = min(config.max_points_ph, len(X))
    tda_idx = sample_indices_balanced(labels, tda_size, config.random_state + 29)
    tda_points_input = X[tda_idx]
    input_thresh = adaptive_edge_length(tda_points_input, quantile=0.35)
    input_dgms = ripser(tda_points_input, maxdim=1, thresh=input_thresh)["dgms"]

    history_rows: List[Dict[str, Any]] = []

    for epoch in range(config.epochs + 1):
        model.train()

        if epoch > 0:
            optimizer.zero_grad()
            z, recon = model(x_tensor)
            loss_recon = F.mse_loss(recon, x_tensor)
            loss_lap = laplacian_penalty(z, adjacency) if lambda_laplacian > 0 else torch.tensor(0.0, device=device)
            loss_st = stress_penalty(z[ref_idx_t], d_ref) if lambda_stress > 0 else torch.tensor(0.0, device=device)
            total_loss = loss_recon + lambda_laplacian * loss_lap + lambda_stress * loss_st
            total_loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                z, recon = model(x_tensor)
                loss_recon = F.mse_loss(recon, x_tensor)
                loss_lap = laplacian_penalty(z, adjacency) if lambda_laplacian > 0 else torch.tensor(0.0, device=device)
                loss_st = stress_penalty(z[ref_idx_t], d_ref) if lambda_stress > 0 else torch.tensor(0.0, device=device)
                total_loss = loss_recon + lambda_laplacian * loss_lap + lambda_stress * loss_st

        if epoch % config.checkpoint_every == 0 or epoch == config.epochs:
            model.eval()
            with torch.no_grad():
                z_np = model.encode(x_tensor).cpu().numpy()
            z_tda = z_np[tda_idx]
            topo_stats, dgms = summarize_latent_topology(z_tda, reference_dgms=input_dgms)
            history_rows.append(
                {
                    "model": model_name,
                    "epoch": int(epoch),
                    "reconstruction_loss": float(loss_recon.detach().cpu().item()),
                    "laplacian_loss": float(loss_lap.detach().cpu().item()),
                    "stress_loss": float(loss_st.detach().cpu().item()),
                    "total_loss": float(total_loss.detach().cpu().item()),
                    "trustworthiness": compute_trustworthiness(X[tda_idx], z_tda, n_neighbors=min(10, len(z_tda) - 1)),
                    **topo_stats,
                }
            )

    history = pd.DataFrame(history_rows)
    return model, history


def plot_latent_space(
    points_2d: np.ndarray,
    labels: Sequence[str],
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    labels = pd.Series(labels).fillna("unknown").astype(str)
    unique_labels = sorted(labels.unique().tolist())
    cmap = plt.cm.get_cmap("tab10", max(1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            points_2d[mask, 0],
            points_2d[mask, 1],
            s=34,
            alpha=0.82,
            label=label,
            color=cmap(idx),
            edgecolors="white",
            linewidths=0.4,
        )

    ax.set_title(title)
    ax.set_xlabel("Latent axis 1")
    ax.set_ylabel("Latent axis 2")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, path)


def collapse_persistence_image(image_tensor: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_tensor, dtype=float)
    if arr.ndim == 4:
        return arr.sum(axis=0)
    if arr.ndim == 3:
        if arr.shape[0] <= 4:
            return arr.sum(axis=0)
        return arr
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported persistence image shape: {arr.shape}")


def plot_topology_evolution(history_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharex=True)
    for model_name, df in history_df.groupby("model"):
        axes[0].plot(df["epoch"], df["h1_bottleneck_to_reference"], marker="o", linewidth=2.0, label=model_name)
        axes[1].plot(df["epoch"], df["trustworthiness"], marker="o", linewidth=2.0, label=model_name)

    axes[0].set_title("Figure 5.2C. H1 bottleneck drift")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Distance to input topology")
    axes[0].grid(alpha=0.25, linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_title("Figure 5.2C. Trustworthiness")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Trustworthiness")
    axes[1].grid(alpha=0.25, linewidth=0.5)
    axes[1].legend(frameon=False, fontsize=8)

    save_figure(fig, path)


def plot_persistence_image_comparison(before_img: np.ndarray, after_img: np.ndarray, path: Path) -> None:
    diff = after_img - before_img
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))

    im0 = axes[0].imshow(before_img, origin="lower", aspect="auto")
    axes[0].set_title("Pre-training reference")
    axes[0].set_xlabel("Birth bins")
    axes[0].set_ylabel("Persistence bins")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(after_img, origin="lower", aspect="auto")
    axes[1].set_title("Topology-regularized latent")
    axes[1].set_xlabel("Birth bins")
    axes[1].set_ylabel("Persistence bins")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    vmax = np.max(np.abs(diff)) if np.isfinite(diff).any() else 1.0
    vmax = max(vmax, 1e-6)
    im2 = axes[2].imshow(diff, origin="lower", aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    axes[2].set_title("Difference (after - before)")
    axes[2].set_xlabel("Birth bins")
    axes[2].set_ylabel("Persistence bins")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Figure 5.2D. Before/after persistence image comparison", fontsize=12)
    save_figure(fig, path)


def run_section_5_2(
    config: PipelineConfig,
    X_windows: np.ndarray,
    meta_windows: pd.DataFrame,
) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(config.section_5_2_dir)
    manifest: List[Dict[str, Any]] = []

    labels = meta_windows["trajectory_group"].astype(str).fillna("unknown").values
    idx = sample_indices_balanced(labels, config.max_windows_for_latent, config.random_state + 7)
    X = X_windows[idx]
    meta = meta_windows.iloc[idx].reset_index(drop=True)
    labels_sub = meta["trajectory_group"].astype(str).tolist()

    before_reference = PCA(n_components=max(3, config.latent_dim), random_state=config.random_state).fit_transform(X)
    before_path = out_dir / "figure_5_2a_pretraining_reference_manifold.png"
    plot_latent_space(
        before_reference[:, :2],
        labels_sub,
        "Figure 5.2A. Pre-training reference manifold (PCA)",
        before_path,
    )
    manifest.append(
        {
            "section": "5.2",
            "type": "figure",
            "file": str(before_path),
            "purpose": "Reference manifold before learning a topology-regularized latent representation.",
        }
    )

    baseline_model, baseline_history = train_autoencoder(
        X=X,
        labels=labels_sub,
        config=config,
        lambda_laplacian=0.0,
        lambda_stress=0.0,
        model_name="baseline_autoencoder",
    )
    topo_model, topo_history = train_autoencoder(
        X=X,
        labels=labels_sub,
        config=config,
        lambda_laplacian=config.lambda_laplacian,
        lambda_stress=config.lambda_stress,
        model_name="topology_regularized_autoencoder",
    )

    x_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        baseline_z = baseline_model.encode(x_tensor).cpu().numpy()
        topo_z = topo_model.encode(x_tensor).cpu().numpy()

    after_path = out_dir / "figure_5_2b_topology_regularized_latent_manifold.png"
    plot_latent_space(
        topo_z[:, :2],
        labels_sub,
        "Figure 5.2B. Topology-regularized latent manifold",
        after_path,
    )
    manifest.append(
        {
            "section": "5.2",
            "type": "figure",
            "file": str(after_path),
            "purpose": "Latent representation after topology-aware regularization.",
        }
    )

    history_df = pd.concat([baseline_history, topo_history], ignore_index=True)
    evolution_path = out_dir / "figure_5_2c_latent_topology_evolution.png"
    plot_topology_evolution(history_df, evolution_path)
    manifest.append(
        {
            "section": "5.2",
            "type": "figure",
            "file": str(evolution_path),
            "purpose": "Evolution of topology drift and trustworthiness during training.",
        }
    )

    compare_idx = sample_indices_balanced(labels_sub, min(config.max_points_ph, len(X)), config.random_state + 99)
    before_sample = before_reference[compare_idx, :3]
    baseline_sample = baseline_z[compare_idx, :3]
    after_sample = topo_z[compare_idx, :3]

    topo_threshold = adaptive_edge_length(np.vstack([before_sample, after_sample]), quantile=0.35)
    topo_batch = np.stack([before_sample, after_sample], axis=0)
    topo_diag_batch = compute_vr_diagrams_for_batch(topo_batch, max_edge_length=topo_threshold)

    baseline_threshold = adaptive_edge_length(np.vstack([before_sample, baseline_sample]), quantile=0.35)
    baseline_batch = np.stack([before_sample, baseline_sample], axis=0)
    baseline_diag_batch = compute_vr_diagrams_for_batch(baseline_batch, max_edge_length=baseline_threshold)

    pi = PersistenceImage(sigma=0.08, n_bins=config.n_bins_persistence_image)
    pi_images = pi.fit_transform(topo_diag_batch)

    before_img = collapse_persistence_image(pi_images[0])
    after_img = collapse_persistence_image(pi_images[1])

    pi_path = out_dir / "figure_5_2d_before_after_persistence_image_comparison.png"
    plot_persistence_image_comparison(before_img, after_img, pi_path)
    manifest.append(
        {
            "section": "5.2",
            "type": "figure",
            "file": str(pi_path),
            "purpose": "Visual comparison of before/after persistence signatures through persistence images.",
        }
    )

    pairwise_landscape = PairwiseDistance(metric="landscape", order=2.0)
    pairwise_betti = PairwiseDistance(metric="betti", order=2.0)
    landscape_dist_baseline = float(pairwise_landscape.fit_transform(baseline_diag_batch)[0, 1])
    betti_dist_baseline = float(pairwise_betti.fit_transform(baseline_diag_batch)[0, 1])
    landscape_dist_topo = float(pairwise_landscape.fit_transform(topo_diag_batch)[0, 1])
    betti_dist_topo = float(pairwise_betti.fit_transform(topo_diag_batch)[0, 1])

    input_reference_dgms = ripser(X[compare_idx], maxdim=1, thresh=adaptive_edge_length(X[compare_idx]))["dgms"]
    baseline_summary, _ = summarize_latent_topology(baseline_z[compare_idx], reference_dgms=input_reference_dgms)
    topo_summary, _ = summarize_latent_topology(topo_z[compare_idx], reference_dgms=input_reference_dgms)

    model_summary_rows = [
        {
            "model": "baseline_autoencoder",
            "final_reconstruction_loss": float(baseline_history["reconstruction_loss"].iloc[-1]),
            "final_trustworthiness": float(baseline_history["trustworthiness"].iloc[-1]),
            "landscape_distance_to_pretraining_reference": landscape_dist_baseline,
            "betti_distance_to_pretraining_reference": betti_dist_baseline,
            **baseline_summary,
        },
        {
            "model": "topology_regularized_autoencoder",
            "final_reconstruction_loss": float(topo_history["reconstruction_loss"].iloc[-1]),
            "final_trustworthiness": float(topo_history["trustworthiness"].iloc[-1]),
            "landscape_distance_to_pretraining_reference": landscape_dist_topo,
            "betti_distance_to_pretraining_reference": betti_dist_topo,
            **topo_summary,
        },
    ]
    model_summary = pd.DataFrame(model_summary_rows)
    model_summary_path = out_dir / "table_5_2_model_summary.csv"
    model_summary.to_csv(model_summary_path, index=False)
    manifest.append(
        {
            "section": "5.2",
            "type": "table",
            "file": str(model_summary_path),
            "purpose": "Final latent-space and topology metrics for baseline and regularized models.",
        }
    )

    checkpoint_history_path = out_dir / "table_5_2_checkpoint_history.csv"
    history_df.to_csv(checkpoint_history_path, index=False)
    manifest.append(
        {
            "section": "5.2",
            "type": "table",
            "file": str(checkpoint_history_path),
            "purpose": "Checkpoint history for losses, trustworthiness, and topology drift.",
        }
    )

    return manifest


# ---------------------------------------------------------------------
# Section 5.3: topological recommendation maps and higher-order graphs
# ---------------------------------------------------------------------

def estimate_dbscan_eps(X: np.ndarray) -> float:
    if len(X) < 3:
        return 0.5
    nn_model = NearestNeighbors(n_neighbors=min(3, len(X))).fit(X)
    dists, _ = nn_model.kneighbors(X)
    eps = float(np.quantile(dists[:, -1], 0.75))
    return max(eps, 0.10)


def build_mapper_graph(concept_features: pd.DataFrame, random_state: int) -> Tuple[nx.Graph, Dict[str, Any]]:
    X = concept_features[
        [
            "usage",
            "success_rate",
            "failure_rate",
            "mean_difficulty",
            "mean_hints",
            "mean_dwell_time",
            "pagerank",
            "betweenness",
            "local_clustering",
            "priority_score",
        ]
    ].to_numpy(dtype=float)

    X = StandardScaler().fit_transform(X)
    lens = PCA(n_components=2, random_state=random_state).fit_transform(X)

    if not KMAP_AVAILABLE:
        raise ImportError("KeplerMapper (kmapper) is required for Section 5.3. Install it before running the script.")

    mapper = km.KeplerMapper(verbose=0)
    graph_dict = mapper.map(
        lens,
        X,
        cover=km.Cover(n_cubes=5, perc_overlap=0.35),
        clusterer=DBSCAN(eps=estimate_dbscan_eps(lens), min_samples=2),
    )

    G = nx.Graph()
    node_members = graph_dict.get("nodes", {})
    for node_id, members in node_members.items():
        members = list(members)
        node_frame = concept_features.iloc[members]
        label_values = node_frame["concept_id"].astype(str).tolist()
        short_label = ", ".join(label_values[:3]) + ("…" if len(label_values) > 3 else "")
        G.add_node(
            node_id,
            size=len(members),
            priority=float(node_frame["priority_score"].mean()),
            failure_rate=float(node_frame["failure_rate"].mean()),
            label=short_label,
            members=label_values,
        )

    links = graph_dict.get("links", {})
    if isinstance(links, dict):
        for source, targets in links.items():
            for target in targets:
                if source != target:
                    G.add_edge(source, target)
    elif isinstance(links, list):
        for edge in links:
            if len(edge) >= 2:
                G.add_edge(edge[0], edge[1])

    return G, graph_dict


def plot_mapper_graph(mapper_graph: nx.Graph, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    if mapper_graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Mapper graph is empty.", ha="center", va="center", fontsize=12)
        ax.axis("off")
        save_figure(fig, path)
        return

    pos = nx.spring_layout(mapper_graph, seed=17, k=0.8 / max(1, math.sqrt(mapper_graph.number_of_nodes())))
    node_sizes = minmax_scale_array([mapper_graph.nodes[n].get("size", 1) for n in mapper_graph.nodes], 350, 1800)
    node_values = np.asarray([mapper_graph.nodes[n].get("priority", 0.0) for n in mapper_graph.nodes], dtype=float)

    nodes = nx.draw_networkx_nodes(
        mapper_graph,
        pos,
        node_size=node_sizes,
        node_color=node_values,
        cmap="viridis",
        linewidths=0.8,
        edgecolors="black",
        ax=ax,
    )
    nx.draw_networkx_edges(mapper_graph, pos, width=1.8, alpha=0.65, ax=ax)

    labels = {n: mapper_graph.nodes[n].get("label", str(n)) for n in mapper_graph.nodes}
    if mapper_graph.number_of_nodes() <= 20:
        nx.draw_networkx_labels(mapper_graph, pos, labels=labels, font_size=8, ax=ax)

    cbar = fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Average recommendation priority")
    ax.set_title("Figure 5.3A. Topological knowledge map (Mapper)")
    ax.axis("off")
    save_figure(fig, path)


def prune_graph_for_plot(graph: nx.DiGraph, concept_features: pd.DataFrame, top_k_nodes: int = 12) -> nx.DiGraph:
    important = concept_features.head(top_k_nodes)["concept_id"].astype(str).tolist()
    subgraph = graph.subgraph(important).copy()
    if subgraph.number_of_edges() == 0 and graph.number_of_edges() > 0:
        edges = sorted(graph.edges(data=True), key=lambda x: x[2].get("weight", 0.0), reverse=True)[:max(10, top_k_nodes)]
        nodes = sorted({u for u, _, _ in edges} | {v for _, v, _ in edges})
        subgraph = graph.subgraph(nodes).copy()
    return subgraph


def plot_learning_graph(graph: nx.DiGraph, concept_features: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Learning graph is empty.", ha="center", va="center", fontsize=12)
        ax.axis("off")
        save_figure(fig, path)
        return

    feature_map = concept_features.set_index("concept_id")
    pos = nx.spring_layout(graph, seed=23, k=1.05 / max(1, math.sqrt(graph.number_of_nodes())))

    node_sizes = []
    node_values = []
    for node in graph.nodes:
        if node in feature_map.index:
            node_sizes.append(float(feature_map.loc[node, "usage"]))
            node_values.append(float(feature_map.loc[node, "failure_rate"]))
        else:
            node_sizes.append(1.0)
            node_values.append(0.0)

    node_sizes = minmax_scale_array(node_sizes, 450, 2200)
    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_values,
        cmap="plasma",
        linewidths=0.8,
        edgecolors="black",
        ax=ax,
    )

    edge_weights = [float(graph[u][v].get("weight", 1.0)) for u, v in graph.edges]
    widths = minmax_scale_array(edge_weights, 0.8, 4.5)
    nx.draw_networkx_edges(
        graph,
        pos,
        width=widths,
        alpha=0.62,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)

    cbar = fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Failure rate")
    ax.set_title("Figure 5.3B. Learning graph for adaptive recommendation")
    ax.axis("off")
    save_figure(fig, path)


def run_section_5_3(
    config: PipelineConfig,
    interactions: pd.DataFrame,
    concept_edges: pd.DataFrame,
) -> List[Dict[str, Any]]:
    out_dir = ensure_dir(config.section_5_3_dir)
    manifest: List[Dict[str, Any]] = []

    graph, concept_summary = build_transition_graph(interactions, concept_edges)
    concept_features = build_concept_feature_table(graph, concept_summary)

    mapper_graph, graph_dict = build_mapper_graph(concept_features, random_state=config.random_state)
    mapper_path = out_dir / "figure_5_3a_topological_knowledge_map.png"
    plot_mapper_graph(mapper_graph, mapper_path)
    manifest.append(
        {
            "section": "5.3",
            "type": "figure",
            "file": str(mapper_path),
            "purpose": "Mapper-based topological summary of concept neighborhoods and recommendation zones.",
        }
    )

    pruned = prune_graph_for_plot(graph, concept_features, top_k_nodes=12)
    learning_graph_path = out_dir / "figure_5_3b_learning_graph.png"
    plot_learning_graph(pruned, concept_features, learning_graph_path)
    manifest.append(
        {
            "section": "5.3",
            "type": "figure",
            "file": str(learning_graph_path),
            "purpose": "Directed learning graph showing strong concept transitions and high-risk nodes.",
        }
    )

    undirected_pruned = pruned.to_undirected()
    cycle_rank = undirected_pruned.number_of_edges() - undirected_pruned.number_of_nodes() + nx.number_connected_components(undirected_pruned) if undirected_pruned.number_of_nodes() > 0 else 0
    triangle_count = sum(nx.triangles(undirected_pruned).values()) // 3 if undirected_pruned.number_of_nodes() > 0 else 0

    top_bridge = concept_features.sort_values("betweenness", ascending=False)["concept_id"].astype(str).head(5).tolist()
    top_priority = concept_features.sort_values("priority_score", ascending=False)["concept_id"].astype(str).head(5).tolist()

    sc_dim = float("nan")
    if TOPONETX_AVAILABLE and undirected_pruned.number_of_nodes() > 0:
        try:
            sc = graph_to_clique_complex(undirected_pruned, max_rank=2)
            sc_dim = float(getattr(sc, "dim", np.nan))
        except Exception:
            sc_dim = float("nan")

    summary_df = pd.DataFrame(
        [
            {
                "n_concepts": int(graph.number_of_nodes()),
                "n_transitions": int(graph.number_of_edges()),
                "pruned_nodes": int(pruned.number_of_nodes()),
                "pruned_edges": int(pruned.number_of_edges()),
                "mapper_nodes": int(mapper_graph.number_of_nodes()),
                "mapper_edges": int(mapper_graph.number_of_edges()),
                "graph_density": float(nx.density(undirected_pruned)) if undirected_pruned.number_of_nodes() > 1 else 0.0,
                "graph_cycle_rank_beta1": int(cycle_rank),
                "n_triangles_2simplices": int(triangle_count),
                "toponetx_complex_dim": sc_dim,
                "top_bridge_concepts": "; ".join(top_bridge),
                "top_priority_concepts": "; ".join(top_priority),
            }
        ]
    )
    summary_path = out_dir / "table_5_3_graph_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    manifest.append(
        {
            "section": "5.3",
            "type": "table",
            "file": str(summary_path),
            "purpose": "Graph, Mapper, and higher-order topology summary for recommendation diagnostics.",
        }
    )

    alerts = concept_features.copy()
    # Simple editorial rule-based action labels
    conditions = [
        (alerts["failure_rate"] >= alerts["failure_rate"].quantile(0.75)) & (alerts["betweenness"] >= alerts["betweenness"].quantile(0.60)),
        alerts["failure_rate"] >= alerts["failure_rate"].quantile(0.75),
        alerts["betweenness"] >= alerts["betweenness"].quantile(0.75),
        alerts["priority_score"] >= alerts["priority_score"].quantile(0.75),
    ]
    choices = [
        "Targeted prerequisite support",
        "Tutoring reinforcement",
        "Bridge concept monitoring",
        "Adaptive sequencing review",
    ]
    alerts["recommended_action"] = np.select(conditions, choices, default="Standard monitoring")
    alerts_table = (
        alerts[
            [
                "concept_id",
                "usage",
                "success_rate",
                "failure_rate",
                "pagerank",
                "betweenness",
                "priority_score",
                "recommended_action",
            ]
        ]
        .sort_values("priority_score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    alerts_path = out_dir / "table_5_3_recommendation_alerts.csv"
    alerts_table.to_csv(alerts_path, index=False)
    manifest.append(
        {
            "section": "5.3",
            "type": "table",
            "file": str(alerts_path),
            "purpose": "Priority concepts and recommended educational interventions derived from graph topology.",
        }
    )

    graph_dict_path = out_dir / "mapper_graph_raw.json"
    with open(graph_dict_path, "w", encoding="utf-8") as f:
        json.dump(graph_dict, f, ensure_ascii=False, indent=2)
    manifest.append(
        {
            "section": "5.3",
            "type": "data",
            "file": str(graph_dict_path),
            "purpose": "Raw Mapper graph for further customization or interactive visualization.",
        }
    )

    return manifest


# ---------------------------------------------------------------------
# Manifest and main
# ---------------------------------------------------------------------

def save_manifest(config: PipelineConfig, entries: List[Dict[str, Any]], data_mode: str) -> None:
    manifest_df = pd.DataFrame(entries)
    manifest_path = config.output_dir / "output_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    metadata = {
        "data_mode": data_mode,
        "config": {
            **asdict(config),
            "input_dir": str(config.input_dir),
            "output_dir": str(config.output_dir),
        },
        "libraries": {
            "gudhi": True,
            "ripser": True,
            "persim": True,
            "giotto_tda": False,
            "manual_tda_replacements": True,
            "kmapper": KMAP_AVAILABLE,
            "toponetx": TOPONETX_AVAILABLE,
            "torch": TORCH_AVAILABLE,
        },
    }
    with open(config.output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Generate compact topological figures and tables for chapter sections 5.1, 5.2, and 5.3."
    )
    parser.add_argument("--input_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs_topological_chapter")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--window_stride", type=int, default=4)
    parser.add_argument("--min_window_events", type=int, default=8)
    parser.add_argument("--max_points_ph", type=int, default=220)
    parser.add_argument("--max_windows_for_latent", type=int, default=480)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--checkpoint_every", type=int, default=6)
    parser.add_argument("--latent_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--k_neighbors", type=int, default=8)
    parser.add_argument("--lambda_laplacian", type=float, default=0.03)
    parser.add_argument("--lambda_stress", type=float, default=0.02)
    parser.add_argument("--n_bins_betti", type=int, default=72)
    parser.add_argument("--n_bins_persistence_image", type=int, default=32)
    parser.add_argument("--disable_synthetic_fallback", action="store_true")

    args = parser.parse_args()
    return PipelineConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        random_state=args.random_state,
        window_size=args.window_size,
        window_stride=args.window_stride,
        min_window_events=args.min_window_events,
        max_points_ph=args.max_points_ph,
        max_windows_for_latent=args.max_windows_for_latent,
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        k_neighbors=args.k_neighbors,
        lambda_laplacian=args.lambda_laplacian,
        lambda_stress=args.lambda_stress,
        n_bins_betti=args.n_bins_betti,
        n_bins_persistence_image=args.n_bins_persistence_image,
        use_synthetic_if_missing=not args.disable_synthetic_fallback,
    )


def main() -> None:
    config = parse_args()
    ensure_dir(config.output_dir)
    ensure_dir(config.section_5_1_dir)
    ensure_dir(config.section_5_2_dir)
    ensure_dir(config.section_5_3_dir)

    set_seed(config.random_state)

    data = load_or_generate_data(config)
    data_mode = "synthetic" if "ability" in data["learner_profiles"].columns and config.use_synthetic_if_missing and not (config.input_dir / "interactions.csv").exists() else "real_or_harmonized"

    interactions = data["interactions"].copy()
    resources = data["resources"].copy()
    concept_edges = data["concept_edges"].copy()
    learner_profiles = data["learner_profiles"].copy()

    window_df = build_learner_windows(
        interactions=interactions,
        resources=resources,
        learner_profiles=learner_profiles,
        window_size=config.window_size,
        window_stride=config.window_stride,
        min_window_events=config.min_window_events,
    )

    if window_df.empty:
        raise RuntimeError("No learner windows could be created. Check the input data or reduce window constraints.")

    X_windows, meta_windows, feature_cols = prepare_window_matrix(window_df)

    manifest_entries: List[Dict[str, Any]] = []
    manifest_entries.extend(run_section_5_1(config, window_df, X_windows, meta_windows))
    manifest_entries.extend(run_section_5_2(config, X_windows, meta_windows))
    manifest_entries.extend(run_section_5_3(config, interactions, concept_edges))

    save_manifest(config, manifest_entries, data_mode=data_mode)

    print("Topological pipeline completed.")
    print(f"Outputs written to: {config.output_dir.resolve()}")
    print(f"Feature count used in learner windows: {len(feature_cols)}")


if __name__ == "__main__":
    main()
