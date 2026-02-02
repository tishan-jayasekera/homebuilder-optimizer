"""
Phase 4 - Referral Ecosystem Clustering
"""
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict

try:
    from community import community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    community_louvain = None


def _merge_small_clusters(G, initial_partition, target_max_clusters=15):
    """Merge small clusters into nearest connected cluster."""
    partition = dict(initial_partition)
    
    while True:
        clusters = defaultdict(list)
        for node, cid in partition.items():
            clusters[cid].append(node)
        
        n_clusters = len(clusters)
        if n_clusters <= target_max_clusters:
            break
        
        cluster_degrees = {}
        for cid, nodes in clusters.items():
            degs = dict(G.degree(nodes, weight="weight"))
            cluster_degrees[cid] = sum(degs.values())
        
        candidate_cids = [cid for cid, deg in cluster_degrees.items() if deg > 0]
        if not candidate_cids:
            break
        
        def cluster_key(cid):
            return (len(clusters[cid]), cluster_degrees[cid])
        
        smallest_cid = min(candidate_cids, key=cluster_key)
        nodes_small = clusters[smallest_cid]
        
        connectivity = defaultdict(float)
        for u in nodes_small:
            if u not in G:
                continue
            for v, data in G[u].items():
                if v not in partition:
                    continue
                cid_v = partition[v]
                if cid_v == smallest_cid:
                    continue
                connectivity[cid_v] += data.get("weight", 1.0)
        
        if not connectivity:
            del cluster_degrees[smallest_cid]
            if not cluster_degrees:
                break
            continue
        
        target_cid = max(connectivity.keys(), key=lambda c: connectivity[c])
        for n in nodes_small:
            partition[n] = target_cid
    
    # Relabel 1..K
    clusters_final = defaultdict(list)
    for node, cid in partition.items():
        clusters_final[cid].append(node)
    
    sorted_cids = sorted(clusters_final.keys())
    cid_map = {old: i + 1 for i, old in enumerate(sorted_cids)}
    return {node: cid_map[cid] for node, cid in partition.items()}


def run_referral_clustering(
    events: pd.DataFrame,
    basis: str = "RefDate",
    min_edge_weight: int = 1,
    min_degree: int = 1,
    resolution: float = 1.5,
    target_max_clusters: int = 15
) -> dict:
    """
    Run Phase 4 referral ecosystem clustering.
    
    Referral definition (commercial):
    - is_referral == True OR
    - MediaPayer_BuilderRegionKey != Dest_BuilderRegionKey
    
    Returns dict with:
        - edges_raw: all edges
        - edges_clean: pruned edges with cluster IDs
        - builder_master: builder-level cluster assignments
        - cluster_summary: cluster-level summary
        - graph: NetworkX graph object
    """
    df = events.copy()
    
    required = ["LeadId", "Dest_BuilderRegionKey", "MediaPayer_BuilderRegionKey", "is_referral"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    # Referral definition
    mask_referral = df["is_referral"].fillna(False).astype(bool)
    mask_cross_payer = (
        df["MediaPayer_BuilderRegionKey"].notna() &
        df["Dest_BuilderRegionKey"].notna() &
        (df["MediaPayer_BuilderRegionKey"] != df["Dest_BuilderRegionKey"])
    )
    
    df_ref = df.loc[mask_referral | mask_cross_payer].copy()
    
    if df_ref.empty:
        return {"edges_raw": pd.DataFrame(), "edges_clean": pd.DataFrame(),
                "builder_master": pd.DataFrame(), "cluster_summary": pd.DataFrame(), "graph": nx.Graph()}
    
    # Build edges
    df_ref = df_ref[
        df_ref["MediaPayer_BuilderRegionKey"].notna() &
        df_ref["Dest_BuilderRegionKey"].notna()
    ]
    df_ref["Origin_builder"] = df_ref["MediaPayer_BuilderRegionKey"]
    df_ref["Dest_builder"] = df_ref["Dest_BuilderRegionKey"]
    
    # Remove self-loops
    df_ref = df_ref[df_ref["Origin_builder"] != df_ref["Dest_builder"]]
    
    if df_ref.empty:
        return {"edges_raw": pd.DataFrame(), "edges_clean": pd.DataFrame(),
                "builder_master": pd.DataFrame(), "cluster_summary": pd.DataFrame(), "graph": nx.Graph()}
    
    edges_raw = (
        df_ref.groupby(["Origin_builder", "Dest_builder"], as_index=False)["LeadId"]
        .nunique()
        .rename(columns={"LeadId": "Referrals"})
    )
    
    # Prune edges
    edges_clean = edges_raw[edges_raw["Referrals"] >= min_edge_weight].copy()
    
    if edges_clean.empty:
        return {"edges_raw": edges_raw, "edges_clean": pd.DataFrame(),
                "builder_master": pd.DataFrame(), "cluster_summary": pd.DataFrame(), "graph": nx.Graph()}
    
    # Build graph and prune by degree
    G_tmp = nx.Graph()
    for _, row in edges_clean.iterrows():
        o, d, w = row["Origin_builder"], row["Dest_builder"], float(row["Referrals"])
        if G_tmp.has_edge(o, d):
            G_tmp[o][d]["weight"] += w
        else:
            G_tmp.add_edge(o, d, weight=w)
    
    degrees = dict(G_tmp.degree(weight="weight"))
    keep_nodes = [n for n, deg in degrees.items() if deg >= min_degree]
    
    edges_clean = edges_clean[
        edges_clean["Origin_builder"].isin(keep_nodes) &
        edges_clean["Dest_builder"].isin(keep_nodes)
    ]
    
    if edges_clean.empty:
        return {"edges_raw": edges_raw, "edges_clean": pd.DataFrame(),
                "builder_master": pd.DataFrame(), "cluster_summary": pd.DataFrame(), "graph": nx.Graph()}
    
    # Build final graph
    G = nx.Graph()
    for _, row in edges_clean.iterrows():
        o, d, w = row["Origin_builder"], row["Dest_builder"], float(row["Referrals"])
        if G.has_edge(o, d):
            G[o][d]["weight"] += w
        else:
            G.add_edge(o, d, weight=w)
    
    # Clustering
    if HAS_LOUVAIN:
        initial_partition = community_louvain.best_partition(
            G, weight="weight", resolution=resolution, random_state=42
        )
    else:
        comps = list(nx.connected_components(G))
        initial_partition = {node: i + 1 for i, comp in enumerate(comps) for node in comp}
    
    partition = _merge_small_clusters(G, initial_partition, target_max_clusters)
    
    # Add cluster IDs to edges
    edges_clean["Cluster_origin"] = edges_clean["Origin_builder"].map(partition)
    edges_clean["Cluster_dest"] = edges_clean["Dest_builder"].map(partition)
    
    # Builder master
    in_flows = (
        edges_clean.groupby("Dest_builder", as_index=False)["Referrals"]
        .sum()
        .rename(columns={"Dest_builder": "BuilderRegionKey", "Referrals": "Referrals_in"})
    )
    out_flows = (
        edges_clean.groupby("Origin_builder", as_index=False)["Referrals"]
        .sum()
        .rename(columns={"Origin_builder": "BuilderRegionKey", "Referrals": "Referrals_out"})
    )
    
    builders = pd.DataFrame({"BuilderRegionKey": sorted(G.nodes())})
    builders["ClusterId"] = builders["BuilderRegionKey"].map(partition)
    builders = builders.merge(in_flows, on="BuilderRegionKey", how="left")
    builders = builders.merge(out_flows, on="BuilderRegionKey", how="left")
    builders[["Referrals_in", "Referrals_out"]] = builders[["Referrals_in", "Referrals_out"]].fillna(0)
    
    def classify_role(row):
        rin, rout = row["Referrals_in"], row["Referrals_out"]
        if rout > 0 and rin == 0:
            return "pure_sender"
        if rin > 0 and rout == 0:
            return "pure_receiver"
        if rin > 0 and rout > 0:
            return "mixed"
        return "isolated"
    
    builders["Role"] = builders.apply(classify_role, axis=1)
    
    # Cluster summary
    cluster_summary = (
        builders.groupby("ClusterId", as_index=False)
        .agg(
            N_builders=("BuilderRegionKey", "nunique"),
            Total_referrals_in=("Referrals_in", "sum"),
            Total_referrals_out=("Referrals_out", "sum")
        )
        .sort_values("ClusterId")
    )
    
    return {
        "edges_raw": edges_raw,
        "edges_clean": edges_clean,
        "builder_master": builders,
        "cluster_summary": cluster_summary,
        "graph": G
    }


def compute_network_metrics(G: nx.Graph, builders_df: pd.DataFrame) -> pd.DataFrame:
    """Compute network metrics: PageRank, ERS, Referral Gap, RMS."""
    builders = builders_df.copy()
    
    if len(G.nodes) == 0:
        for col in ["deg", "pagerank", "ERS", "Referral_Gap", "RMS"]:
            builders[col] = 0.0
        return builders
    
    pr = nx.pagerank(G, weight="weight")
    deg = dict(G.degree(weight="weight"))
    
    builders["deg"] = builders["BuilderRegionKey"].map(deg).fillna(0)
    builders["pagerank"] = builders["BuilderRegionKey"].map(pr).fillna(0.0)
    
    total_ref_in = builders["Referrals_in"].sum()
    builders["ERS"] = builders["pagerank"] * total_ref_in if total_ref_in else 0.0
    builders["Referral_Gap"] = builders["Referrals_in"] - builders["ERS"]
    
    # RMS - referral multiplier score
    if "ROAS" in builders.columns:
        roas_map = builders.set_index("BuilderRegionKey")["ROAS"].to_dict()
        rms_vals = []
        for node in builders["BuilderRegionKey"]:
            if node not in G:
                rms_vals.append(0.0)
                continue
            s = 0.0
            for nbr, attrs in G[node].items():
                w = attrs.get("weight", 0)
                r = roas_map.get(nbr, 0.0)
                s += w * r
            rms_vals.append(s)
        builders["RMS"] = rms_vals
    else:
        builders["RMS"] = 0.0
    
    return builders
