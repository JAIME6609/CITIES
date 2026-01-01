# -*- coding: utf-8 -*-
"""
EUS Bayesian Network + Exploratory Urban Analytics (FULL ARTIFACT EXPORT VERSION)

Goal
-----
This script generates (or ingests) an illustrative urban dataset, produces the full set of
tables and figures in a strict, publication-oriented order, and exports everything to a
single results folder on the Windows Desktop:
  * One Excel workbook with multiple sheets (tables/results in order)
  * One figures folder with PNGs (figures in order)
  * Optional model artifacts (trained neural network)

Design principles (reviewer-facing)
-----------------------------------
* Operational, reproducible outputs: fixed seeds, explicit parameters, full artifact manifest.
* Decision-relevant discretizations: categorical states (Low/Medium/High etc.) are explicit.
* Probabilistic interpretation: BN edges indicate conditional dependencies, not causal ID unless stated.
* Auditability: model checks, state inventories, CPD normalization diagnostics, and inference outputs saved.

Dependencies
------------
pip install numpy pandas matplotlib seaborn scikit-learn pgmpy openpyxl xlsxwriter tensorflow

Note: TensorFlow is optional; if unavailable, the script will skip the neural-network block
and still export all other outputs.
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import random
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.decomposition import PCA

# Bayesian Network (pgmpy)
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Network visualization
import networkx as nx

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    seed: int = 42
    n_samples: int = 1000

    # Output
    desktop_dir: str = os.path.join(os.path.expanduser("~"), "Desktop")
    output_folder_prefix: str = "EUS_BN_FULL_RESULTS"
    timestamp_folder: bool = True

    # CO2 (synthetic units)
    co2_min: int = 50
    co2_max: int = 800

    # Clustering
    k_min: int = 2
    k_max: int = 9
    kmeans_random_state: int = 42
    pca_components: int = 3

    # Neural net
    run_neural_net: bool = True
    test_size: float = 0.20
    nn_random_state: int = 42


CFG = Config()


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_output_root(cfg: Config) -> str:
    base = os.path.join(cfg.desktop_dir, cfg.output_folder_prefix)
    if cfg.timestamp_folder:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{base}_{stamp}"
    _safe_mkdir(base)
    return base


def save_png(fig_path: str, dpi: int = 220) -> None:
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def fig_name(out_fig_dir: str, order: int, label: str) -> str:
    # Ensures strict sort order in file browsers.
    return os.path.join(out_fig_dir, f"{order:02d}_{label}.png")


def write_df_sheet(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame, index: bool = True) -> None:
    # Excel sheet name must be <= 31 chars.
    safe = sheet_name[:31]
    df.to_excel(writer, sheet_name=safe, index=index)


def as_single_column_df(lines: List[str], col_name: str = "Value") -> pd.DataFrame:
    return pd.DataFrame({col_name: lines})


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# DATA GENERATION (SYNTHETIC, ILLUSTRATIVE)
# =============================================================================

def generate_synthetic_dataset(cfg: Config) -> pd.DataFrame:
    """
    Generates an illustrative (non-empirical) dataset. Each row represents an urban unit.
    The variable set is designed to align with a proximity/accessibility framing and
    to support probabilistic scenario reasoning in a BN.
    """
    set_global_seeds(cfg.seed)
    n = cfg.n_samples

    df = pd.DataFrame({
        "Services_Distribution": np.random.choice(["High", "Medium", "Low"], n, p=[0.3, 0.5, 0.2]),
        "Services_Proximity": np.random.choice(["High", "Medium", "Low"], n),
        "Services_Coverage": np.random.choice(["High", "Medium", "Low"], n),
        "Bike_Lanes_Available": np.random.choice(["Yes", "No"], n, p=[0.55, 0.45]),
        "STU": np.random.choice(["Frequent", "Moderate", "Low"], n),
        "Motorized_Vehicle_Usage": np.random.choice(["High", "Medium", "Low"], n),
        "CO2_Emissions": np.random.randint(cfg.co2_min, cfg.co2_max + 1, n),
        "Distance_Reduction": np.random.choice(["Yes", "No"], n),
        "Free_Time": np.random.choice(["High", "Medium", "Low"], n),
        "Pedestrian_Infrastructure": np.random.choice(["Adequate", "Inadequate"], n),
        "Pedestrian_Areas": np.random.choice(["Present", "Absent"], n),
        "Quality_of_Life": np.random.choice(["High", "Medium", "Low"], n),
        "Public_Health": np.random.choice(["Good", "Fair", "Poor"], n),
    })

    return df


# =============================================================================
# DESCRIPTIVE TABLES
# =============================================================================

def frequency_tables(df: pd.DataFrame, cols: List[str]) -> Dict[str, pd.DataFrame]:
    out = {}
    for c in cols:
        vc = df[c].value_counts(dropna=False).rename_axis(c).reset_index(name="Count")
        vc["Proportion"] = vc["Count"] / vc["Count"].sum()
        out[c] = vc
    return out


# =============================================================================
# BAYESIAN NETWORK CONSTRUCTION (DISCRETE, INTERPRETIVE)
# =============================================================================

def build_bayesian_network() -> Tuple[DiscreteBayesianNetwork, Dict[str, List[str]]]:
    """
    Constructs a discrete Bayesian network for scenario inference.
    Edges encode conditional dependencies that are interpretable for planning,
    but they do not constitute causal identification unless an explicit strategy exists.
    """
    model = DiscreteBayesianNetwork([
        ("Services_Distribution", "Services_Proximity"),
        ("Services_Proximity", "Services_Coverage"),
        ("Services_Coverage", "STU"),
        ("Bike_Lanes_Available", "STU"),
        ("STU", "Motorized_Vehicle_Usage"),
        ("STU", "Pedestrian_Infrastructure"),
        ("Pedestrian_Infrastructure", "Pedestrian_Areas"),
        ("Distance_Reduction", "Free_Time"),
        ("Pedestrian_Areas", "Quality_of_Life"),
        ("Free_Time", "Quality_of_Life"),
        ("Quality_of_Life", "Public_Health"),
        ("Motorized_Vehicle_Usage", "CO2_Emissions"),
        ("Distance_Reduction", "CO2_Emissions"),
    ])

    # State dictionaries (explicit, reviewer-auditable)
    state_names = {
        "Services_Distribution": ["High", "Medium", "Low"],
        "Services_Proximity": ["High", "Medium", "Low"],
        "Services_Coverage": ["High", "Medium", "Low"],
        "Bike_Lanes_Available": ["Yes", "No"],
        "STU": ["Frequent", "Moderate", "Low"],
        "Motorized_Vehicle_Usage": ["High", "Medium", "Low"],
        "Distance_Reduction": ["Yes", "No"],
        "Pedestrian_Infrastructure": ["Adequate", "Inadequate"],
        "Pedestrian_Areas": ["Present", "Absent"],
        "Free_Time": ["High", "Medium", "Low"],
        "Quality_of_Life": ["High", "Medium", "Low"],
        "Public_Health": ["Good", "Fair", "Poor"],
        "CO2_Emissions": ["Low", "Medium", "High"],  # discretized CO2 states for BN
    }

    # Priors
    cpd_services_dist = TabularCPD(
        variable="Services_Distribution",
        variable_card=3,
        values=[[0.30], [0.50], [0.20]],
        state_names={"Services_Distribution": state_names["Services_Distribution"]},
    )

    cpd_bike = TabularCPD(
        variable="Bike_Lanes_Available",
        variable_card=2,
        values=[[0.55], [0.45]],
        state_names={"Bike_Lanes_Available": state_names["Bike_Lanes_Available"]},
    )

    cpd_dist_reduction = TabularCPD(
        variable="Distance_Reduction",
        variable_card=2,
        values=[[0.50], [0.50]],
        state_names={"Distance_Reduction": state_names["Distance_Reduction"]},
    )

    # Services_Proximity | Services_Distribution
    cpd_services_prox = TabularCPD(
        variable="Services_Proximity",
        variable_card=3,
        values=[
            # Dist: High, Medium, Low
            [0.60, 0.40, 0.20],  # Prox High
            [0.30, 0.40, 0.40],  # Prox Medium
            [0.10, 0.20, 0.40],  # Prox Low
        ],
        evidence=["Services_Distribution"],
        evidence_card=[3],
        state_names={
            "Services_Proximity": state_names["Services_Proximity"],
            "Services_Distribution": state_names["Services_Distribution"],
        },
    )

    # Services_Coverage | Services_Proximity
    cpd_services_cov = TabularCPD(
        variable="Services_Coverage",
        variable_card=3,
        values=[
            # Prox: High, Medium, Low
            [0.65, 0.40, 0.20],  # Coverage High
            [0.25, 0.40, 0.40],  # Coverage Medium
            [0.10, 0.20, 0.40],  # Coverage Low
        ],
        evidence=["Services_Proximity"],
        evidence_card=[3],
        state_names={
            "Services_Coverage": state_names["Services_Coverage"],
            "Services_Proximity": state_names["Services_Proximity"],
        },
    )

    # STU | Services_Coverage, Bike_Lanes_Available
    # Columns: (High,Yes), (High,No), (Med,Yes), (Med,No), (Low,Yes), (Low,No)
    cpd_stu = TabularCPD(
        variable="STU",
        variable_card=3,
        values=[
            [0.65, 0.45, 0.45, 0.25, 0.25, 0.10],  # Frequent
            [0.25, 0.35, 0.35, 0.40, 0.35, 0.30],  # Moderate
            [0.10, 0.20, 0.20, 0.35, 0.40, 0.60],  # Low
        ],
        evidence=["Services_Coverage", "Bike_Lanes_Available"],
        evidence_card=[3, 2],
        state_names={
            "STU": state_names["STU"],
            "Services_Coverage": state_names["Services_Coverage"],
            "Bike_Lanes_Available": state_names["Bike_Lanes_Available"],
        },
    )

    # Motorized_Vehicle_Usage | STU
    cpd_motor = TabularCPD(
        variable="Motorized_Vehicle_Usage",
        variable_card=3,
        values=[
            # STU: Frequent, Moderate, Low
            [0.20, 0.40, 0.70],  # High motorized use
            [0.30, 0.35, 0.20],  # Medium
            [0.50, 0.25, 0.10],  # Low motorized use
        ],
        evidence=["STU"],
        evidence_card=[3],
        state_names={
            "Motorized_Vehicle_Usage": state_names["Motorized_Vehicle_Usage"],
            "STU": state_names["STU"],
        },
    )

    # Pedestrian_Infrastructure | STU
    cpd_ped_infra = TabularCPD(
        variable="Pedestrian_Infrastructure",
        variable_card=2,
        values=[
            # STU: Frequent, Moderate, Low
            [0.80, 0.50, 0.20],  # Adequate
            [0.20, 0.50, 0.80],  # Inadequate
        ],
        evidence=["STU"],
        evidence_card=[3],
        state_names={
            "Pedestrian_Infrastructure": state_names["Pedestrian_Infrastructure"],
            "STU": state_names["STU"],
        },
    )

    # Pedestrian_Areas | Pedestrian_Infrastructure
    cpd_ped_areas = TabularCPD(
        variable="Pedestrian_Areas",
        variable_card=2,
        values=[
            # Infra: Adequate, Inadequate
            [0.75, 0.30],  # Present
            [0.25, 0.70],  # Absent
        ],
        evidence=["Pedestrian_Infrastructure"],
        evidence_card=[2],
        state_names={
            "Pedestrian_Areas": state_names["Pedestrian_Areas"],
            "Pedestrian_Infrastructure": state_names["Pedestrian_Infrastructure"],
        },
    )

    # Free_Time | Distance_Reduction
    cpd_free = TabularCPD(
        variable="Free_Time",
        variable_card=3,
        values=[
            # Dist.Reduction: Yes, No
            [0.60, 0.20],  # High
            [0.30, 0.40],  # Medium
            [0.10, 0.40],  # Low
        ],
        evidence=["Distance_Reduction"],
        evidence_card=[2],
        state_names={
            "Free_Time": state_names["Free_Time"],
            "Distance_Reduction": state_names["Distance_Reduction"],
        },
    )

    # Quality_of_Life | Pedestrian_Areas, Free_Time
    # Columns: (Present,High), (Present,Medium), (Present,Low), (Absent,High), (Absent,Medium), (Absent,Low)
    cpd_qol = TabularCPD(
        variable="Quality_of_Life",
        variable_card=3,
        values=[
            [0.80, 0.60, 0.30, 0.50, 0.30, 0.10],  # High
            [0.15, 0.30, 0.40, 0.35, 0.40, 0.30],  # Medium
            [0.05, 0.10, 0.30, 0.15, 0.30, 0.60],  # Low
        ],
        evidence=["Pedestrian_Areas", "Free_Time"],
        evidence_card=[2, 3],
        state_names={
            "Quality_of_Life": state_names["Quality_of_Life"],
            "Pedestrian_Areas": state_names["Pedestrian_Areas"],
            "Free_Time": state_names["Free_Time"],
        },
    )

    # Public_Health | Quality_of_Life
    cpd_health = TabularCPD(
        variable="Public_Health",
        variable_card=3,
        values=[
            # QoL: High, Medium, Low
            [0.70, 0.40, 0.20],  # Good
            [0.25, 0.40, 0.35],  # Fair
            [0.05, 0.20, 0.45],  # Poor
        ],
        evidence=["Quality_of_Life"],
        evidence_card=[3],
        state_names={
            "Public_Health": state_names["Public_Health"],
            "Quality_of_Life": state_names["Quality_of_Life"],
        },
    )

    # CO2_Emissions | Motorized_Vehicle_Usage, Distance_Reduction
    # Columns: (High,Yes), (High,No), (Med,Yes), (Med,No), (Low,Yes), (Low,No)
    cpd_co2 = TabularCPD(
        variable="CO2_Emissions",
        variable_card=3,
        values=[
            [0.20, 0.35, 0.15, 0.25, 0.05, 0.10],  # Low
            [0.35, 0.40, 0.45, 0.45, 0.25, 0.30],  # Medium
            [0.45, 0.25, 0.40, 0.30, 0.70, 0.60],  # High
        ],
        evidence=["Motorized_Vehicle_Usage", "Distance_Reduction"],
        evidence_card=[3, 2],
        state_names={
            "CO2_Emissions": state_names["CO2_Emissions"],
            "Motorized_Vehicle_Usage": state_names["Motorized_Vehicle_Usage"],
            "Distance_Reduction": state_names["Distance_Reduction"],
        },
    )

    model.add_cpds(
        cpd_services_dist, cpd_bike, cpd_dist_reduction,
        cpd_services_prox, cpd_services_cov, cpd_stu,
        cpd_motor, cpd_ped_infra, cpd_ped_areas,
        cpd_free, cpd_qol, cpd_health, cpd_co2
    )

    return model, state_names


def bn_inventory_table(model: DiscreteBayesianNetwork, state_names: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for node in model.nodes():
        parents = list(model.get_parents(node))
        rows.append({
            "Node": node,
            "Num_States": len(state_names.get(node, [])),
            "States": ", ".join(state_names.get(node, [])),
            "Parents": ", ".join(parents) if parents else "",
            "Num_Parents": len(parents),
        })
    return pd.DataFrame(rows).sort_values(["Node"]).reset_index(drop=True)


def bn_check_report(model: DiscreteBayesianNetwork) -> Tuple[bool, str]:
    try:
        ok = model.check_model()
        return bool(ok), "check_model() completed successfully: CPDs are normalized, consistent, and state cardinalities match."
    except Exception as e:
        return False, f"check_model() failed: {repr(e)}"


def bn_posterior_table(
    infer: VariableElimination,
    query_var: str,
    evidence: Dict[str, str],
    state_order: List[str],
) -> pd.DataFrame:
    q = infer.query(variables=[query_var], evidence=evidence, show_progress=False)
    probs = q.values.tolist()
    return pd.DataFrame({
        query_var: state_order,
        "Posterior_Probability": probs
    })


# =============================================================================
# MAIN PIPELINE (EXPORT EVERYTHING IN ORDER)
# =============================================================================

def run_pipeline(cfg: Config) -> None:
    set_global_seeds(cfg.seed)

    out_root = build_output_root(cfg)
    out_fig = os.path.join(out_root, "figures_png")
    out_rep = os.path.join(out_root, "excel_reports")
    out_mod = os.path.join(out_root, "models_optional")
    _safe_mkdir(out_fig)
    _safe_mkdir(out_rep)
    _safe_mkdir(out_mod)

    excel_path = os.path.join(out_rep, "RESULTS_ALL_TABLES.xlsx")
    manifest_rows = []

    # -------------------------------------------------------------------------
    # PART 1: DATASET + DESCRIPTIVE TABLES
    # -------------------------------------------------------------------------
    df = generate_synthetic_dataset(cfg)

    # Table 3.1: df.head()
    t31 = df.head(10).copy()

    # Table 3.2: describe(CO2)
    t32 = df["CO2_Emissions"].describe().to_frame(name="CO2_Emissions").reset_index().rename(columns={"index": "Statistic"})

    # Table 3.3: frequency tables for key categorical vars
    freq_cols = ["STU", "Quality_of_Life", "Pedestrian_Infrastructure", "Bike_Lanes_Available", "Motorized_Vehicle_Usage"]
    freq_map = frequency_tables(df, freq_cols)

    # Figures 3.1 & 3.2
    plt.figure(figsize=(10, 6))
    plt.hist(df["CO2_Emissions"], bins=20, alpha=0.75)
    plt.title("Distribution of CO₂ Emissions")
    plt.xlabel("CO₂ (simulated units)")
    plt.ylabel("Frequency")
    f31 = fig_name(out_fig, 1, "Figure_3_1_Histogram_CO2")
    save_png(f31)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(df["CO2_Emissions"], fill=True, linewidth=2)
    plt.title("Estimated Density of CO₂ Emissions (KDE)")
    plt.xlabel("CO₂ (simulated units)")
    f32 = fig_name(out_fig, 2, "Figure_3_2_KDE_CO2")
    save_png(f32)

    # Boxplots / countplots (Figures 3.3–3.7)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Motorized_Vehicle_Usage", y="CO2_Emissions")
    plt.title("CO₂ Emissions by Motorized Vehicle Usage")
    plt.xlabel("Motorized vehicle usage")
    plt.ylabel("CO₂ (simulated units)")
    f33 = fig_name(out_fig, 3, "Figure_3_3_Boxplot_CO2_by_Motorized_Usage")
    save_png(f33)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="STU", y="CO2_Emissions")
    plt.title("CO₂ Emissions by Sustainable Transport Usage (STU)")
    plt.xlabel("Sustainable transport usage (STU)")
    plt.ylabel("CO₂ (simulated units)")
    f34 = fig_name(out_fig, 4, "Figure_3_4_Boxplot_CO2_by_STU")
    save_png(f34)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Pedestrian_Areas", y="CO2_Emissions")
    plt.title("CO₂ Emissions by Presence of Pedestrian Areas")
    plt.xlabel("Pedestrian areas")
    plt.ylabel("CO₂ (simulated units)")
    f35 = fig_name(out_fig, 5, "Figure_3_5_Boxplot_CO2_by_Pedestrian_Areas")
    save_png(f35)

    plt.figure(figsize=(9, 6))
    sns.countplot(data=df, x="Pedestrian_Infrastructure", hue="Quality_of_Life")
    plt.title("Quality of Life by Pedestrian Infrastructure")
    plt.xlabel("Pedestrian infrastructure")
    plt.ylabel("Frequency")
    f36 = fig_name(out_fig, 6, "Figure_3_6_Countplot_QoL_by_Ped_Infrastructure")
    save_png(f36)

    plt.figure(figsize=(9, 6))
    sns.countplot(data=df, x="STU", hue="Public_Health")
    plt.title("Public Health by Sustainable Transport Usage")
    plt.xlabel("Sustainable transport usage (STU)")
    plt.ylabel("Frequency")
    f37 = fig_name(out_fig, 7, "Figure_3_7_Countplot_PublicHealth_by_STU")
    save_png(f37)

    # Correlation heatmap + pairplot (Figures 3.12–3.13; stored later in strict numbering)
    df_numeric = pd.get_dummies(df, drop_first=True)
    corr = df_numeric.corr(numeric_only=True)

    # -------------------------------------------------------------------------
    # PART 2: BAYESIAN NETWORK (STRUCTURE, CHECKS, INFERENCE)
    # -------------------------------------------------------------------------
    model, state_names = build_bayesian_network()

    # Result 3.1: check_model()
    ok, msg = bn_check_report(model)
    r31 = as_single_column_df([
        f"Model check ok: {ok}",
        msg
    ], col_name="BN_Check_Report")

    # Table 3.4: inventory of nodes/states/parents
    t34 = bn_inventory_table(model, state_names)

    # Figure 3.8: network structure
    G = nx.DiGraph()
    G.add_edges_from(model.edges())
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=cfg.seed)
    nx.draw_networkx_nodes(G, pos, node_size=1200, alpha=0.95)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=16, width=1.8, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title("Bayesian Network Structure (Conditional Dependencies)")
    plt.axis("off")
    f38 = fig_name(out_fig, 8, "Figure_3_8_Bayesian_Network_Structure")
    save_png(f38)

    # Table 3.5: posterior for STU with evidence
    infer = VariableElimination(model)
    evidence = {"Services_Coverage": "High", "Bike_Lanes_Available": "Yes"}
    t35 = bn_posterior_table(infer, "STU", evidence=evidence, state_order=state_names["STU"])
    t35_evidence = pd.DataFrame({
        "Evidence_Variable": list(evidence.keys()),
        "Evidence_State": list(evidence.values()),
    })

    # -------------------------------------------------------------------------
    # PART 3: CLUSTERING + PCA (ROBUSTNESS/STRUCTURE EXPLORATION)
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    # Elbow method (Figure 3.14 in the requested ordering earlier; here kept strictly ordered as a figure artifact)
    inertias = []
    K_range = list(range(cfg.k_min, cfg.k_max + 1))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=cfg.kmeans_random_state, n_init=10)
        km.fit(df_scaled)
        inertias.append(km.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, "o-", linewidth=2)
    plt.title("Elbow Method (K-Means)")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    f39 = fig_name(out_fig, 9, "Figure_3_14_Elbow_Method_KMeans")
    save_png(f39)

    # Silhouette analysis table (saved)
    sil_scores = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=cfg.kmeans_random_state, n_init=10)
        labels = km.fit_predict(df_scaled)
        score = silhouette_score(df_scaled, labels)
        sil_scores.append(score)
    t_sil = pd.DataFrame({"K": K_range, "Silhouette_Score": sil_scores}).sort_values("K").reset_index(drop=True)
    best_k = int(t_sil.loc[t_sil["Silhouette_Score"].idxmax(), "K"])

    # Fit best K and attach cluster labels
    kmeans = KMeans(n_clusters=best_k, random_state=cfg.kmeans_random_state, n_init=10)
    df["Cluster"] = kmeans.fit_predict(df_scaled)

    # PCA
    pca = PCA(n_components=cfg.pca_components, random_state=cfg.seed)
    pcs = pca.fit_transform(df_scaled)
    df["PCA1"] = pcs[:, 0]
    df["PCA2"] = pcs[:, 1]
    df["PCA3"] = pcs[:, 2]
    t_pca_var = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(cfg.pca_components)],
        "Explained_Variance_Ratio": pca.explained_variance_ratio_.tolist(),
        "Cumulative_Explained_Variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
    })

    # Figure 3.15: PCA 2D
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10")
    plt.title("Clusters in PCA Space (2D)")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    f40 = fig_name(out_fig, 10, "Figure_3_15_PCA_2D_by_Cluster")
    save_png(f40)

    # PCA 3D (Figure 3.15 variant; saved as separate artifact)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(df["PCA1"], df["PCA2"], df["PCA3"], c=df["Cluster"], cmap="tab10", s=45, alpha=0.9)
    ax.set_title("Clusters in PCA Space (3D)")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.10)
    f41 = fig_name(out_fig, 11, "Figure_3_15_PCA_3D_by_Cluster")
    plt.tight_layout()
    plt.savefig(f41, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Table 3.8: cluster interpretive summaries
    t38_mean_co2 = df.groupby("Cluster")[["CO2_Emissions"]].mean().reset_index()
    t38_stu = (
        df.groupby(["Cluster"])["STU"]
        .value_counts()
        .rename("Count")
        .reset_index()
        .sort_values(["Cluster", "Count"], ascending=[True, False])
    )
    t38_qol = (
        df.groupby(["Cluster"])["Quality_of_Life"]
        .value_counts()
        .rename("Count")
        .reset_index()
        .sort_values(["Cluster", "Count"], ascending=[True, False])
    )
    t38_ped = (
        df.groupby(["Cluster"])["Pedestrian_Infrastructure"]
        .value_counts()
        .rename("Count")
        .reset_index()
        .sort_values(["Cluster", "Count"], ascending=[True, False])
    )

    # Heatmap + pairplot (dense; recommended as appendix artifacts)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, cbar=True)
    plt.title("Correlation Heatmap (One-Hot Encoded Variables)")
    f42 = fig_name(out_fig, 12, "Figure_3_12_Correlation_Heatmap_Encoded")
    save_png(f42)

    # Pairplot can be heavy; keep to a manageable subset
    pair_cols = ["CO2_Emissions", "PCA1", "PCA2", "PCA3", "Cluster"]
    g = sns.pairplot(df[pair_cols], hue="Cluster", diag_kind="kde")
    f43 = fig_name(out_fig, 13, "Figure_3_13_Pairplot_CO2_PCA_Cluster")
    g.fig.savefig(f43, dpi=220, bbox_inches="tight")
    plt.close(g.fig)

    # -------------------------------------------------------------------------
    # PART 4: NEURAL NETWORK BASELINE (OPTIONAL, BUT INCLUDED IF TF AVAILABLE)
    # -------------------------------------------------------------------------
    nn_artifacts = {
        "ran_neural_net": False,
        "test_loss": None,
        "test_accuracy": None
    }
    t36 = pd.DataFrame()
    t37 = pd.DataFrame()
    t_history = pd.DataFrame()

    if cfg.run_neural_net:
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from sklearn.utils.class_weight import compute_class_weight

            # TF reproducibility
            tf.random.set_seed(cfg.seed)

            # Features: all except target
            target = "Quality_of_Life"
            X = df.drop(columns=[target]).copy()

            # Encode categoricals (one-hot)
            X_enc = pd.get_dummies(X, drop_first=True)

            # Target encoding
            le = LabelEncoder()
            y = le.fit_transform(df[target].values)

            X_train, X_test, y_train, y_test = train_test_split(
                X_enc, y, test_size=cfg.test_size, random_state=cfg.nn_random_state, stratify=y
            )

            scaler_nn = StandardScaler()
            X_train_scaled = scaler_nn.fit_transform(X_train)
            X_test_scaled = scaler_nn.transform(X_test)

            classes = np.unique(y_train)
            cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
            class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

            model_dl = Sequential([
                Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
                BatchNormalization(),
                Dropout(0.30),

                Dense(64, activation="relu"),
                BatchNormalization(),
                Dropout(0.25),

                Dense(32, activation="relu"),
                Dropout(0.15),

                Dense(len(le.classes_), activation="softmax"),
            ])

            model_dl.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
                verbose=0
            )

            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-5,
                verbose=0
            )

            history = model_dl.fit(
                X_train_scaled,
                y_train,
                epochs=200,
                batch_size=32,
                validation_split=0.20,
                callbacks=[early_stop, reduce_lr],
                class_weight=class_weight_dict,
                verbose=0
            )

            # Training curves (Figures 3.9–3.10)
            t_history = pd.DataFrame(history.history)

            plt.figure(figsize=(10, 6))
            plt.plot(history.history["loss"], label="Training loss")
            plt.plot(history.history["val_loss"], label="Validation loss")
            plt.title("Loss evolution (improved model)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            f44 = fig_name(out_fig, 14, "Figure_3_9_Loss_Curve_Train_vs_Val")
            save_png(f44)

            plt.figure(figsize=(10, 6))
            plt.plot(history.history["accuracy"], label="Training accuracy")
            plt.plot(history.history["val_accuracy"], label="Validation accuracy")
            plt.title("Accuracy evolution (improved model)")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            f45 = fig_name(out_fig, 15, "Figure_3_10_Accuracy_Curve_Train_vs_Val")
            save_png(f45)

            # Test set performance (Tables 3.6–3.7)
            test_loss, test_acc = model_dl.evaluate(X_test_scaled, y_test, verbose=0)
            t36 = pd.DataFrame([{
                "Test_Loss": float(test_loss),
                "Test_Accuracy": float(test_acc),
                "Test_Size": int(len(y_test)),
                "Train_Size": int(len(y_train)),
                "Num_Features": int(X_train_scaled.shape[1]),
            }])

            # Predictions
            y_pred_proba = model_dl.predict(X_test_scaled, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)

            rep = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            t37 = pd.DataFrame(rep).transpose().reset_index().rename(columns={"index": "Class_or_Aggregate"})

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(7, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
            disp.plot(values_format="d")
            plt.title("Confusion Matrix — Improved Quality of Life Model")
            f46 = fig_name(out_fig, 16, "Figure_3_11_Confusion_Matrix_QoL_Model")
            plt.tight_layout()
            plt.savefig(f46, dpi=220, bbox_inches="tight")
            plt.close()

            # Save model artifacts
            model_path = os.path.join(out_mod, "Quality_of_Life_Model.keras")
            model_dl.save(model_path)

            # Export encoders/scalers metadata
            meta = {
                "label_classes": le.classes_.tolist(),
                "n_features": int(X_train_scaled.shape[1]),
                "feature_columns": X_enc.columns.tolist(),
                "class_weight": class_weight_dict,
                "seed": cfg.seed,
            }
            with open(os.path.join(out_mod, "Quality_of_Life_Model_Metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            nn_artifacts["ran_neural_net"] = True
            nn_artifacts["test_loss"] = float(test_loss)
            nn_artifacts["test_accuracy"] = float(test_acc)

        except Exception as e:
            nn_artifacts["ran_neural_net"] = False
            nn_artifacts["error"] = repr(e)

    # -------------------------------------------------------------------------
    # EXPORT: ONE EXCEL WORKBOOK (SHEETS IN ORDER) + MANIFEST
    # -------------------------------------------------------------------------
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        # 3.1.1 Tables (order)
        write_df_sheet(writer, "T3_1_df_head", t31, index=False)
        manifest_rows.append({"Order": 1, "Type": "Table", "Name": "Table 3.1 df.head()", "Location": "Excel:T3_1_df_head"})

        write_df_sheet(writer, "T3_2_CO2_describe", t32, index=False)
        manifest_rows.append({"Order": 2, "Type": "Table", "Name": "Table 3.2 CO2 describe()", "Location": "Excel:T3_2_CO2_describe"})

        # Frequency tables (each as a separate sheet)
        for i, col in enumerate(freq_cols, start=1):
            sheet = f"T3_3_freq_{col}"[:31]
            write_df_sheet(writer, sheet, freq_map[col], index=False)
            manifest_rows.append({"Order": 2 + i, "Type": "Table", "Name": f"Table 3.3 Frequencies: {col}", "Location": f"Excel:{sheet}"})

        # 3.1.2 BN checks + inventory
        write_df_sheet(writer, "R3_1_BN_check", r31, index=False)
        manifest_rows.append({"Order": 10, "Type": "Result", "Name": "Result 3.1 BN check_model()", "Location": "Excel:R3_1_BN_check"})

        write_df_sheet(writer, "T3_4_BN_inventory", t34, index=False)
        manifest_rows.append({"Order": 11, "Type": "Table", "Name": "Table 3.4 BN node/state inventory", "Location": "Excel:T3_4_BN_inventory"})

        # 3.1.3 BN posterior baseline
        write_df_sheet(writer, "T3_5_STU_posterior", t35, index=False)
        write_df_sheet(writer, "T3_5_Evidence", t35_evidence, index=False)
        manifest_rows.append({"Order": 12, "Type": "Table", "Name": "Table 3.5 STU posterior (baseline evidence)", "Location": "Excel:T3_5_STU_posterior + T3_5_Evidence"})

        # Clustering/PCA tables (robustness/structure exploration)
        write_df_sheet(writer, "KMeans_Silhouette", t_sil, index=False)
        write_df_sheet(writer, "PCA_Variance", t_pca_var, index=False)
        write_df_sheet(writer, "T3_8_Cluster_CO2", t38_mean_co2, index=False)
        write_df_sheet(writer, "T3_8_Cluster_STU", t38_stu, index=False)
        write_df_sheet(writer, "T3_8_Cluster_QoL", t38_qol, index=False)
        write_df_sheet(writer, "T3_8_Cluster_PedInf", t38_ped, index=False)

        # Neural net tables (baseline comparison)
        if len(t36) > 0:
            write_df_sheet(writer, "T3_6_Test_Metrics", t36, index=False)
        else:
            write_df_sheet(writer, "T3_6_Test_Metrics", as_single_column_df(
                ["Neural network block was skipped or failed. See Manifest_NN_Status sheet."],
                col_name="Note"
            ), index=False)

        if len(t37) > 0:
            write_df_sheet(writer, "T3_7_Class_Report", t37, index=False)
        else:
            write_df_sheet(writer, "T3_7_Class_Report", as_single_column_df(
                ["Neural network block was skipped or failed. See Manifest_NN_Status sheet."],
                col_name="Note"
            ), index=False)

        if len(t_history) > 0:
            write_df_sheet(writer, "NN_Training_History", t_history, index=False)
        else:
            write_df_sheet(writer, "NN_Training_History", as_single_column_df(
                ["No training history available (NN skipped/failed)."],
                col_name="Note"
            ), index=False)

        # Manifest sheets
        manifest_df = pd.DataFrame(manifest_rows).sort_values("Order").reset_index(drop=True)
        write_df_sheet(writer, "MANIFEST_TablesResults", manifest_df, index=False)

        nn_status = pd.DataFrame([nn_artifacts])
        write_df_sheet(writer, "MANIFEST_NN_Status", nn_status, index=False)

        paths_df = pd.DataFrame([{
            "Output_Root": out_root,
            "Figures_Folder": out_fig,
            "Excel_Folder": out_rep,
            "Excel_Workbook": excel_path,
            "Models_Folder": out_mod,
            "Seed": cfg.seed,
            "N_Samples": cfg.n_samples,
            "Best_K": best_k,
        }])
        write_df_sheet(writer, "MANIFEST_Paths_Params", paths_df, index=False)

        # Save the final dataset (with clusters + PCA) for traceability
        write_df_sheet(writer, "DATASET_Final", df, index=False)

    # -------------------------------------------------------------------------
    # FINAL: WRITE A SHORT TEXT MANIFEST FILE
    # -------------------------------------------------------------------------
    txt_manifest = os.path.join(out_root, "MANIFEST_README.txt")
    with open(txt_manifest, "w", encoding="utf-8") as f:
        f.write("EUS_BN_FULL_RESULTS — Artifact Manifest\n")
        f.write("=================================================\n\n")
        f.write(f"Output root: {out_root}\n")
        f.write(f"Figures (PNG): {out_fig}\n")
        f.write(f"Excel workbook: {excel_path}\n")
        f.write(f"Optional models: {out_mod}\n\n")
        f.write("Figures are numbered to preserve order.\n")
        f.write("Excel sheets preserve table/result ordering and include a manifest.\n\n")
        f.write("Neural net status:\n")
        f.write(json.dumps(nn_artifacts, indent=2))
        f.write("\n")

    print("\nEXPORT COMPLETED SUCCESSFULLY.\n")
    print(f"Results folder created on Desktop:\n  {out_root}\n")
    print("Key exports:")
    print(f"  * Excel workbook: {excel_path}")
    print(f"  * Figures folder: {out_fig}")
    print(f"  * Manifest: {txt_manifest}")


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    # If needed, a user may override Desktop output (e.g., on non-Windows systems)
    # by setting CFG.desktop_dir before running run_pipeline(CFG).
    run_pipeline(CFG)


