"""
Postcode-level referral optimization utilities.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import re

import networkx as nx
import numpy as np
import pandas as pd

from .referral_logic import count_leads_refs, lead_ref_masks, prepare_referral_ids
from .referral_clusters import _merge_small_clusters
from .network_optimization import build_builder_targets, analyze_network_leverage

try:
    from community import community_louvain

    HAS_LOUVAIN = True
except Exception:
    community_louvain = None
    HAS_LOUVAIN = False


POSTCODE_CANDIDATES = ["postcode", "Postcode", "PostCode", "zip_code", "ZipCode"]
LOCALITY_CANDIDATES = ["suburb", "Suburb", "locality", "Locality"]
STATE_CANDIDATES = ["state", "State", "region", "Region"]


def _find_col(columns, candidates):
    cols = {c.lower(): c for c in columns}
    cols_stripped = {c.strip().lower(): c for c in columns}
    cols_norm = {re.sub(r"[^a-z0-9]+", "", c.strip().lower()): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        c = cand.lower()
        if c in cols:
            return cols[c]
        c = cand.strip().lower()
        if c in cols_stripped:
            return cols_stripped[c]
        c = re.sub(r"[^a-z0-9]+", "", cand.strip().lower())
        if c in cols_norm:
            return cols_norm[c]
    return None


def _clean_text_series(series: pd.Series, uppercase: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(dtype="object")
    s = (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "none": np.nan})
    )
    if uppercase:
        s = s.str.upper()
    return s


def _normalize_geo_unit(series: pd.Series, uppercase: bool = True) -> pd.Series:
    s = _clean_text_series(series, uppercase=False)
    is_digits = s.fillna("").str.fullmatch(r"\d{1,6}")
    s.loc[is_digits] = s.loc[is_digits].str.zfill(4)
    if uppercase:
        s = s.str.upper()
    return s


def _derive_region_from_builder_key(series: pd.Series) -> pd.Series:
    """
    Fallback geography derivation when no postcode/suburb columns exist.
    We split `BuilderName - RegionName` and use the right-side region token.
    """
    s = _clean_text_series(series, uppercase=False)
    split = s.str.split(r"\s+-\s+", n=1, expand=True)
    derived = split[1] if isinstance(split, pd.DataFrame) and 1 in split.columns else s
    derived = derived.fillna(s)
    return _normalize_geo_unit(derived, uppercase=True)


def _attach_geo_columns(events_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, Optional[str], Optional[str]]:
    df = events_df.copy()
    dest_col = _find_col(df.columns, ["Dest_BuilderRegionKey"])
    if dest_col is None:
        raise KeyError("Missing required column: Dest_BuilderRegionKey")

    postcode_col = _find_col(df.columns, POSTCODE_CANDIDATES)
    locality_col = _find_col(df.columns, LOCALITY_CANDIDATES)
    state_col = _find_col(df.columns, STATE_CANDIDATES)

    if postcode_col is not None:
        geo = _normalize_geo_unit(df[postcode_col], uppercase=True)
        source = "direct_column"
    elif locality_col is not None:
        geo = _normalize_geo_unit(df[locality_col], uppercase=True)
        source = "direct_column"
    else:
        geo = _derive_region_from_builder_key(df[dest_col])
        source = "derived_from_key"

    state = _normalize_geo_unit(df[state_col], uppercase=True) if state_col else pd.Series(np.nan, index=df.index)

    df["_geo_unit"] = geo
    df["_geo_state"] = state
    df["_geo_source"] = source
    return df, dest_col, postcode_col, locality_col


def _flow_builder_to_postcode_map(mapping_df: pd.DataFrame) -> Dict[str, str]:
    if mapping_df is None or mapping_df.empty:
        return {}
    df = mapping_df.copy()
    if "Event_Count" not in df.columns:
        df["Event_Count"] = 1
    if "Share_of_Builder" not in df.columns:
        totals = df.groupby("BuilderRegionKey")["Event_Count"].transform("sum").replace(0, np.nan)
        df["Share_of_Builder"] = df["Event_Count"] / totals
    df = df.sort_values(["BuilderRegionKey", "Share_of_Builder", "Event_Count"], ascending=[True, False, False])
    top = df.drop_duplicates("BuilderRegionKey")
    return top.set_index("BuilderRegionKey")["Postcode"].to_dict()


def _safe_ref_count(group: pd.DataFrame) -> int:
    if group is None or group.empty:
        return 0
    refs = count_leads_refs(group)[1]
    if refs > 0:
        return int(refs)
    lead_id_col = _find_col(group.columns, ["LeadId", "lead_id", "LeadID"])
    if lead_id_col:
        return int(group[lead_id_col].nunique())
    return int(len(group))


def extract_postcode_mapping(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a mapping between builders and postcodes/suburbs/regions.

    Strategy:
    1) postcode column if present
    2) suburb/locality column if present
    3) split Dest_BuilderRegionKey on " - " and use right-side token
    """
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=["BuilderRegionKey", "Postcode", "Source"])

    df, dest_col, _, _ = _attach_geo_columns(events_df)
    mapping = (
        df[[dest_col, "_geo_unit", "_geo_state"]]
        .dropna(subset=[dest_col, "_geo_unit"])
        .assign(_n=1)
        .groupby([dest_col, "_geo_unit", "_geo_state"], dropna=False)["_n"]
        .sum()
        .reset_index(name="Event_Count")
        .rename(columns={dest_col: "BuilderRegionKey", "_geo_unit": "Postcode", "_geo_state": "State"})
    )
    if mapping.empty:
        return pd.DataFrame(columns=["BuilderRegionKey", "Postcode", "Source"])

    totals = mapping.groupby("BuilderRegionKey")["Event_Count"].transform("sum").replace(0, np.nan)
    mapping["Share_of_Builder"] = mapping["Event_Count"] / totals
    mapping["Source"] = str(df["_geo_source"].iloc[0])
    mapping = mapping.sort_values(["BuilderRegionKey", "Event_Count"], ascending=[True, False])
    mapping["Is_Primary"] = ~mapping.duplicated("BuilderRegionKey")
    return mapping[
        [
            "BuilderRegionKey",
            "Postcode",
            "Source",
            "State",
            "Event_Count",
            "Share_of_Builder",
            "Is_Primary",
        ]
    ]


def build_builder_postcode_coverage(events_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each builder, compute postcode coverage and volume mix.
    """
    if events_df is None or events_df.empty:
        return pd.DataFrame(
            columns=[
                "BuilderRegionKey",
                "Postcode",
                "Lead_Count",
                "Referral_Count",
                "Media_Spend",
                "Share_of_Builder_Volume",
            ]
        )

    df, dest_col, _, _ = _attach_geo_columns(events_df)
    cost_col = _find_col(df.columns, ["MediaCost_referral_event", "MediaCost_builder_touch", "MediaCost_origin_lead", "MediaCost"])
    df, _, _, _ = prepare_referral_ids(df, inplace=False)

    scoped = df.dropna(subset=[dest_col, "_geo_unit"]).copy()
    if scoped.empty:
        return pd.DataFrame(
            columns=[
                "BuilderRegionKey",
                "Postcode",
                "Lead_Count",
                "Referral_Count",
                "Media_Spend",
                "Share_of_Builder_Volume",
            ]
        )

    def _agg(g: pd.DataFrame) -> pd.Series:
        leads, refs, _, _ = count_leads_refs(g)
        spend = pd.to_numeric(g[cost_col], errors="coerce").sum() if cost_col else 0.0
        return pd.Series({"Lead_Count": leads, "Referral_Count": refs, "Media_Spend": float(spend)})

    coverage = (
        scoped.groupby([dest_col, "_geo_unit"], dropna=False)
        .apply(_agg)
        .reset_index()
        .rename(columns={dest_col: "BuilderRegionKey", "_geo_unit": "Postcode"})
    )
    coverage["Total_Volume"] = coverage["Lead_Count"] + coverage["Referral_Count"]
    totals = coverage.groupby("BuilderRegionKey")["Total_Volume"].transform("sum").replace(0, np.nan)
    coverage["Share_of_Builder_Volume"] = coverage["Total_Volume"] / totals
    coverage["Share_of_Builder_Volume"] = coverage["Share_of_Builder_Volume"].fillna(0.0)

    if mapping_df is not None and not mapping_df.empty and "State" in mapping_df.columns:
        state_map = (
            mapping_df.sort_values(["BuilderRegionKey", "Event_Count"], ascending=[True, False])
            .drop_duplicates(["BuilderRegionKey", "Postcode"])
            [["BuilderRegionKey", "Postcode", "State"]]
        )
        coverage = coverage.merge(state_map, on=["BuilderRegionKey", "Postcode"], how="left")

    return coverage[
        [
            "BuilderRegionKey",
            "Postcode",
            "Lead_Count",
            "Referral_Count",
            "Media_Spend",
            "Share_of_Builder_Volume",
        ]
        + (["State"] if "State" in coverage.columns else [])
    ]


def build_postcode_referral_flows(
    events_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    min_flow_threshold: int = 1,
) -> dict:
    """
    Compute postcode-to-postcode referral flows and classify net generators/consumers.
    """
    if events_df is None or events_df.empty:
        return {
            "flows_df": pd.DataFrame(columns=["Origin_Postcode", "Dest_Postcode", "Referral_Count", "Media_Spend"]),
            "postcode_summary": pd.DataFrame(
                columns=[
                    "Postcode",
                    "Referrals_In",
                    "Referrals_Out",
                    "Net_Flow",
                    "Classification",
                    "Builders_Count",
                    "Total_Leads",
                    "Total_Spend",
                    "Avg_CPR",
                ]
            ),
            "graph": nx.DiGraph(),
        }

    df, dest_col, _, _ = _attach_geo_columns(events_df)
    payer_col = _find_col(df.columns, ["MediaPayer_BuilderRegionKey", "_attributed_payer"])
    if payer_col is None:
        return {
            "flows_df": pd.DataFrame(columns=["Origin_Postcode", "Dest_Postcode", "Referral_Count", "Media_Spend"]),
            "postcode_summary": pd.DataFrame(),
            "graph": nx.DiGraph(),
        }

    cost_col = _find_col(df.columns, ["MediaCost_referral_event", "MediaCost_builder_touch", "MediaCost_origin_lead", "MediaCost"])
    builder_to_postcode = _flow_builder_to_postcode_map(mapping_df)

    df, _, _, _ = prepare_referral_ids(df, inplace=False)
    _, _, _, orig_set = count_leads_refs(df)
    lead_mask, ref_mask = lead_ref_masks(df, orig_set)
    cross_mask = (
        df[payer_col].notna()
        & df[dest_col].notna()
        & (df[payer_col].astype(str) != df[dest_col].astype(str))
    )
    flow_mask = ref_mask | cross_mask
    flow_df = df.loc[flow_mask].copy()

    if flow_df.empty:
        coverage = build_builder_postcode_coverage(events_df, mapping_df)
        postcode_base = sorted(set(coverage["Postcode"].dropna().tolist()))
        summary = pd.DataFrame({"Postcode": postcode_base})
        summary["Referrals_In"] = 0
        summary["Referrals_Out"] = 0
        summary["Net_Flow"] = 0
        summary["Classification"] = "isolated"
        if coverage.empty:
            summary["Builders_Count"] = 0
            summary["Total_Leads"] = 0
            summary["Total_Spend"] = 0.0
            summary["Avg_CPR"] = 0.0
        else:
            builders = coverage.groupby("Postcode")["BuilderRegionKey"].nunique()
            leads = coverage.groupby("Postcode")["Lead_Count"].sum()
            spend = coverage.groupby("Postcode")["Media_Spend"].sum()
            refs = coverage.groupby("Postcode")["Referral_Count"].sum().replace(0, np.nan)
            summary["Builders_Count"] = summary["Postcode"].map(builders).fillna(0).astype(int)
            summary["Total_Leads"] = summary["Postcode"].map(leads).fillna(0).astype(int)
            summary["Total_Spend"] = summary["Postcode"].map(spend).fillna(0.0)
            summary["Avg_CPR"] = (summary["Postcode"].map(spend) / summary["Postcode"].map(refs)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return {"flows_df": pd.DataFrame(), "postcode_summary": summary, "graph": nx.DiGraph()}

    flow_df["Origin_Postcode"] = flow_df[payer_col].map(builder_to_postcode)
    flow_df["Dest_Postcode"] = flow_df[dest_col].map(builder_to_postcode)
    flow_df["Origin_Postcode"] = flow_df["Origin_Postcode"].fillna(_derive_region_from_builder_key(flow_df[payer_col]))
    flow_df["Dest_Postcode"] = flow_df["Dest_Postcode"].fillna(_derive_region_from_builder_key(flow_df[dest_col]))
    flow_df = flow_df.dropna(subset=["Origin_Postcode", "Dest_Postcode"])
    flow_df = flow_df[flow_df["Origin_Postcode"] != flow_df["Dest_Postcode"]]

    if flow_df.empty:
        return {
            "flows_df": pd.DataFrame(columns=["Origin_Postcode", "Dest_Postcode", "Referral_Count", "Media_Spend"]),
            "postcode_summary": pd.DataFrame(),
            "graph": nx.DiGraph(),
        }

    def _agg_flow(g: pd.DataFrame) -> pd.Series:
        refs = _safe_ref_count(g)
        spend = pd.to_numeric(g[cost_col], errors="coerce").sum() if cost_col else 0.0
        return pd.Series({"Referral_Count": refs, "Media_Spend": float(spend)})

    flows_df = (
        flow_df.groupby(["Origin_Postcode", "Dest_Postcode"], dropna=False)
        .apply(_agg_flow)
        .reset_index()
    )
    flows_df = flows_df[flows_df["Referral_Count"] >= max(int(min_flow_threshold), 1)]
    flows_df = flows_df.sort_values("Referral_Count", ascending=False).reset_index(drop=True)

    graph = nx.DiGraph()
    for _, row in flows_df.iterrows():
        o = row["Origin_Postcode"]
        d = row["Dest_Postcode"]
        w = float(row["Referral_Count"])
        s = float(row["Media_Spend"])
        if graph.has_edge(o, d):
            graph[o][d]["weight"] += w
            graph[o][d]["media_spend"] += s
        else:
            graph.add_edge(o, d, weight=w, media_spend=s)

    coverage = build_builder_postcode_coverage(events_df, mapping_df)
    postcode_set = set(flows_df["Origin_Postcode"]).union(set(flows_df["Dest_Postcode"]))
    if coverage is not None and not coverage.empty:
        postcode_set = postcode_set.union(set(coverage["Postcode"].dropna().tolist()))
    postcode_base = pd.DataFrame({"Postcode": sorted(postcode_set)})

    inbound = flows_df.groupby("Dest_Postcode")["Referral_Count"].sum()
    outbound = flows_df.groupby("Origin_Postcode")["Referral_Count"].sum()
    postcode_base["Referrals_In"] = postcode_base["Postcode"].map(inbound).fillna(0.0)
    postcode_base["Referrals_Out"] = postcode_base["Postcode"].map(outbound).fillna(0.0)
    postcode_base["Net_Flow"] = postcode_base["Referrals_Out"] - postcode_base["Referrals_In"]

    def _classify(row):
        rin = float(row["Referrals_In"])
        rout = float(row["Referrals_Out"])
        net = float(row["Net_Flow"])
        if rin <= 0 and rout <= 0:
            return "isolated"
        if net > 0:
            return "net_generator"
        if net < 0:
            return "net_consumer"
        return "balanced"

    postcode_base["Classification"] = postcode_base.apply(_classify, axis=1)

    if coverage is not None and not coverage.empty:
        builders = coverage.groupby("Postcode")["BuilderRegionKey"].nunique()
        leads = coverage.groupby("Postcode")["Lead_Count"].sum()
        spend = coverage.groupby("Postcode")["Media_Spend"].sum()
        refs = coverage.groupby("Postcode")["Referral_Count"].sum().replace(0, np.nan)
        postcode_base["Builders_Count"] = postcode_base["Postcode"].map(builders).fillna(0).astype(int)
        postcode_base["Total_Leads"] = postcode_base["Postcode"].map(leads).fillna(0).astype(int)
        postcode_base["Total_Spend"] = postcode_base["Postcode"].map(spend).fillna(0.0)
        postcode_base["Avg_CPR"] = (
            postcode_base["Postcode"].map(spend) / postcode_base["Postcode"].map(refs)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        postcode_base["Builders_Count"] = 0
        postcode_base["Total_Leads"] = 0
        postcode_base["Total_Spend"] = 0.0
        postcode_base["Avg_CPR"] = 0.0

    postcode_summary = postcode_base.sort_values("Net_Flow", ascending=False).reset_index(drop=True)
    return {"flows_df": flows_df, "postcode_summary": postcode_summary, "graph": graph}


def cluster_postcodes(
    flows_df: pd.DataFrame,
    graph: nx.DiGraph,
    resolution: float = 1.0,
    target_max_clusters: int = 12,
) -> dict:
    """
    Cluster postcodes into referral communities using Louvain + small-cluster merging.
    """
    if flows_df is None or flows_df.empty:
        return {
            "postcode_clusters": pd.DataFrame(columns=["Postcode", "ClusterId", "Referrals_In", "Referrals_Out", "Net_Flow"]),
            "cluster_summary": pd.DataFrame(),
            "graph": nx.Graph(),
        }

    undirected = nx.Graph()
    for _, row in flows_df.iterrows():
        o = row["Origin_Postcode"]
        d = row["Dest_Postcode"]
        w = float(row.get("Referral_Count", 0))
        if undirected.has_edge(o, d):
            undirected[o][d]["weight"] += w
        else:
            undirected.add_edge(o, d, weight=w)

    if undirected.number_of_nodes() == 0:
        return {
            "postcode_clusters": pd.DataFrame(columns=["Postcode", "ClusterId", "Referrals_In", "Referrals_Out", "Net_Flow"]),
            "cluster_summary": pd.DataFrame(),
            "graph": undirected,
        }

    if HAS_LOUVAIN and undirected.number_of_edges() > 0:
        initial_partition = community_louvain.best_partition(
            undirected, weight="weight", resolution=float(resolution), random_state=42
        )
    else:
        comps = list(nx.connected_components(undirected))
        initial_partition = {node: i + 1 for i, comp in enumerate(comps) for node in comp}

    partition = _merge_small_clusters(
        undirected, initial_partition=initial_partition, target_max_clusters=max(int(target_max_clusters), 1)
    )

    inbound = flows_df.groupby("Dest_Postcode")["Referral_Count"].sum()
    outbound = flows_df.groupby("Origin_Postcode")["Referral_Count"].sum()
    cluster_nodes = sorted(set(flows_df["Origin_Postcode"]).union(set(flows_df["Dest_Postcode"])))
    postcode_clusters = pd.DataFrame({"Postcode": cluster_nodes})
    postcode_clusters["ClusterId"] = postcode_clusters["Postcode"].map(partition)
    postcode_clusters["Referrals_In"] = postcode_clusters["Postcode"].map(inbound).fillna(0.0)
    postcode_clusters["Referrals_Out"] = postcode_clusters["Postcode"].map(outbound).fillna(0.0)
    postcode_clusters["Net_Flow"] = postcode_clusters["Referrals_Out"] - postcode_clusters["Referrals_In"]

    totals = postcode_clusters.assign(Volume=postcode_clusters["Referrals_In"] + postcode_clusters["Referrals_Out"])

    def _cluster_class(net_val: float) -> str:
        if net_val > 0:
            return "net_generator"
        if net_val < 0:
            return "net_consumer"
        return "balanced"

    cluster_summary = (
        totals.groupby("ClusterId", as_index=False)
        .agg(
            N_Postcodes=("Postcode", "nunique"),
            Total_Referrals_In=("Referrals_In", "sum"),
            Total_Referrals_Out=("Referrals_Out", "sum"),
            Net_Flow=("Net_Flow", "sum"),
        )
        .sort_values("ClusterId")
    )
    cluster_summary["Net_Classification"] = cluster_summary["Net_Flow"].map(_cluster_class)

    top_postcodes = (
        totals.sort_values("Volume", ascending=False)
        .groupby("ClusterId")["Postcode"]
        .apply(lambda s: s.head(5).tolist())
    )
    cluster_summary["Top_Postcodes"] = cluster_summary["ClusterId"].map(top_postcodes).apply(lambda x: x if isinstance(x, list) else [])

    for node, cid in partition.items():
        undirected.nodes[node]["cluster_id"] = int(cid)

    return {"postcode_clusters": postcode_clusters, "cluster_summary": cluster_summary, "graph": undirected}


def calculate_postcode_shortfalls(
    events_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    default_target_per_builder: Optional[float] = None,
) -> pd.DataFrame:
    """
    Allocate builder targets into postcode-level targets and compute postcode gaps.
    """
    if events_df is None or events_df.empty:
        return pd.DataFrame(
            columns=[
                "Postcode",
                "Allocated_Target",
                "Actual_Leads",
                "Actual_Referrals",
                "Shortfall",
                "Surplus",
                "Contributing_Builders",
                "Pct_Behind",
                "Risk_Level",
            ]
        )

    dest_col = _find_col(events_df.columns, ["Dest_BuilderRegionKey"])
    if not dest_col:
        return pd.DataFrame()

    coverage = coverage_df.copy() if coverage_df is not None else pd.DataFrame()
    if coverage.empty:
        coverage = build_builder_postcode_coverage(events_df, mapping_df)
    if coverage.empty:
        return pd.DataFrame()

    targets_df = build_builder_targets(events_df)
    if targets_df is None:
        targets_df = pd.DataFrame()

    if targets_df.empty and default_target_per_builder and default_target_per_builder > 0:
        builders = coverage["BuilderRegionKey"].dropna().unique().tolist()
        targets_df = pd.DataFrame(
            {
                "Builder": builders,
                "LeadTarget": float(default_target_per_builder),
                "JobStart": pd.Timestamp.now() - pd.Timedelta(days=30),
                "JobEnd": pd.Timestamp.now() + pd.Timedelta(days=90),
                "DailyTarget": float(default_target_per_builder) / 120.0,
            }
        )

    if targets_df.empty:
        return pd.DataFrame()

    target_map = (
        targets_df.groupby("Builder", as_index=False)["LeadTarget"]
        .sum()
        .rename(columns={"Builder": "BuilderRegionKey"})
    )

    alloc = coverage.merge(target_map, on="BuilderRegionKey", how="left")
    alloc = alloc.dropna(subset=["LeadTarget"]).copy()
    if alloc.empty:
        return pd.DataFrame()

    alloc["Allocated_Target_part"] = alloc["LeadTarget"] * alloc["Share_of_Builder_Volume"].fillna(0.0)
    alloc["Builder_Postcode_Gap"] = (alloc["Allocated_Target_part"] - alloc["Lead_Count"]).clip(lower=0.0)

    postcode_rollup = (
        alloc.groupby("Postcode", as_index=False)
        .agg(
            Allocated_Target=("Allocated_Target_part", "sum"),
            Actual_Leads=("Lead_Count", "sum"),
            Actual_Referrals=("Referral_Count", "sum"),
            _Builder_Gap=("Builder_Postcode_Gap", "sum"),
        )
    )

    postcode_rollup["Shortfall"] = (postcode_rollup["Allocated_Target"] - postcode_rollup["Actual_Leads"]).clip(lower=0.0)
    postcode_rollup["Surplus"] = (postcode_rollup["Actual_Leads"] - postcode_rollup["Allocated_Target"]).clip(lower=0.0)
    postcode_rollup["Pct_Behind"] = (
        postcode_rollup["Shortfall"] / postcode_rollup["Allocated_Target"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    contrib = (
        alloc[alloc["Builder_Postcode_Gap"] > 0]
        .sort_values(["Postcode", "Builder_Postcode_Gap"], ascending=[True, False])
        .groupby("Postcode")
        .apply(lambda g: [f"{row.BuilderRegionKey} ({row.Builder_Postcode_Gap:,.1f})" for row in g.itertuples()])
    )
    postcode_rollup["Contributing_Builders"] = postcode_rollup["Postcode"].map(contrib).apply(lambda x: x if isinstance(x, list) else [])

    def _risk(row):
        if row["Shortfall"] <= 0 and row["Surplus"] > 0:
            return "surplus"
        if row["Pct_Behind"] >= 0.20:
            return "critical"
        if row["Pct_Behind"] >= 0.08:
            return "at_risk"
        return "on_track"

    postcode_rollup["Risk_Level"] = postcode_rollup.apply(_risk, axis=1)
    return postcode_rollup[
        [
            "Postcode",
            "Allocated_Target",
            "Actual_Leads",
            "Actual_Referrals",
            "Shortfall",
            "Surplus",
            "Contributing_Builders",
            "Pct_Behind",
            "Risk_Level",
        ]
    ].sort_values("Shortfall", ascending=False)


def _top_builders_by_postcode(events_df: pd.DataFrame, mapping_df: pd.DataFrame) -> Dict[str, List[str]]:
    coverage = build_builder_postcode_coverage(events_df, mapping_df)
    if coverage.empty:
        return {}
    top = (
        coverage.sort_values(["Postcode", "Lead_Count", "Referral_Count"], ascending=[True, False, False])
        .groupby("Postcode")["BuilderRegionKey"]
        .apply(lambda s: s.head(5).tolist())
    )
    return top.to_dict()


def build_postcode_optimization_plan(
    shortfalls_df: pd.DataFrame,
    flows_df: pd.DataFrame,
    postcode_summary: pd.DataFrame,
    events_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    budget: float = 50_000,
) -> dict:
    """
    Build budget allocation plan from net-generator postcodes into shortfall postcodes.
    """
    empty = {
        "plan_df": pd.DataFrame(
            columns=[
                "Target_Postcode",
                "Shortfall_Leads",
                "Generator_Postcode",
                "Transfer_Rate",
                "eCPR",
                "Budget_Allocation",
                "Expected_Referrals",
                "Builder_Recommendations",
            ]
        ),
        "cluster_plan_df": pd.DataFrame(),
        "summary": {
            "total_budget_allocated": 0.0,
            "total_expected_referrals": 0.0,
            "coverage_pct": 0.0,
            "blended_ecpr": 0.0,
            "postcodes_covered": 0,
            "postcodes_uncoverable": [],
        },
        "generator_utilization": pd.DataFrame(
            columns=["Generator_Postcode", "Total_Allocated", "Capacity_Used_Pct", "Destinations_Served"]
        ),
    }

    if (
        shortfalls_df is None
        or shortfalls_df.empty
        or flows_df is None
        or flows_df.empty
        or postcode_summary is None
        or postcode_summary.empty
    ):
        return empty

    budget = float(max(budget, 0.0))
    shortfalls = shortfalls_df.copy()
    if "Shortfall" not in shortfalls.columns:
        return empty
    shortfalls = shortfalls[shortfalls["Shortfall"] > 0].sort_values("Shortfall", ascending=False)
    if shortfalls.empty:
        return empty

    summary = postcode_summary.copy()
    if "Net_Flow" not in summary.columns:
        summary["Net_Flow"] = 0.0
    if "Referrals_Out" not in summary.columns:
        summary["Referrals_Out"] = 0.0
    generators = summary[summary["Net_Flow"] > 0]["Postcode"].dropna().astype(str).tolist()
    if not generators:
        empty["summary"]["postcodes_uncoverable"] = shortfalls["Postcode"].astype(str).tolist()
        return empty

    flows = flows_df.copy()
    flows = flows[(flows["Referral_Count"] > 0)].copy()
    if flows.empty:
        return empty

    total_sent = flows.groupby("Origin_Postcode")["Referral_Count"].sum().replace(0, np.nan)
    flows["Transfer_Rate"] = flows["Referral_Count"] / flows["Origin_Postcode"].map(total_sent)
    flows["eCPR"] = (
        pd.to_numeric(flows["Media_Spend"], errors="coerce") / pd.to_numeric(flows["Referral_Count"], errors="coerce")
    ).replace([np.inf, -np.inf], np.nan)

    generator_avg_cpr = summary.set_index("Postcode")["Avg_CPR"].to_dict() if "Avg_CPR" in summary.columns else {}
    flows["eCPR"] = flows["eCPR"].fillna(flows["Origin_Postcode"].map(generator_avg_cpr))
    flows["eCPR"] = flows["eCPR"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    leverage_df = analyze_network_leverage(events_df)
    builder_lookup = _top_builders_by_postcode(events_df, mapping_df)

    capacity_raw = (
        summary.set_index("Postcode")[["Net_Flow", "Referrals_Out"]]
        if set(["Net_Flow", "Referrals_Out"]).issubset(summary.columns)
        else pd.DataFrame()
    )
    capacity = {}
    for gen in generators:
        if gen in capacity_raw.index:
            net = float(capacity_raw.loc[gen, "Net_Flow"])
            out = float(capacity_raw.loc[gen, "Referrals_Out"])
            cap = max(net, out * 0.30, 1.0)
        else:
            out = float(total_sent.get(gen, 0) or 0)
            cap = max(out * 0.30, 1.0)
        capacity[gen] = cap
    capacity_remaining = dict(capacity)

    rows = []
    remaining_budget = budget
    uncovered = []

    for target_row in shortfalls.itertuples():
        target = str(target_row.Postcode)
        shortfall = float(target_row.Shortfall)
        if shortfall <= 0:
            continue

        target_flows = flows[
            (flows["Dest_Postcode"].astype(str) == target)
            & (flows["Origin_Postcode"].astype(str).isin(generators))
        ].copy()
        target_flows = target_flows[target_flows["Transfer_Rate"] > 0]
        target_flows = target_flows[target_flows["eCPR"] > 0]
        if target_flows.empty:
            uncovered.append(target)
            continue

        target_flows = target_flows.sort_values(["eCPR", "Transfer_Rate", "Referral_Count"], ascending=[True, False, False])
        need = shortfall
        allocated_for_target = 0.0

        for c in target_flows.itertuples():
            if need <= 0 or remaining_budget <= 0:
                break
            generator = str(c.Origin_Postcode)
            cap_left = float(capacity_remaining.get(generator, 0.0))
            if cap_left <= 0:
                continue
            ecpr = float(c.eCPR)
            if not np.isfinite(ecpr) or ecpr <= 0:
                continue

            max_from_budget = remaining_budget / ecpr
            expected = min(need, cap_left, max_from_budget)
            if expected <= 0:
                continue

            spend = expected * ecpr
            recs = []
            src_builders = builder_lookup.get(generator, [])
            tgt_builders = builder_lookup.get(target, [])
            if src_builders:
                recs.append(f"Scale spend on {', '.join(src_builders[:3])} in {generator}")
            if tgt_builders:
                recs.append(f"Route to shortfall builders {', '.join(tgt_builders[:3])} in {target}")
            if leverage_df is not None and not leverage_df.empty:
                recs.append("Prioritize historical high-transfer payer/destination pairs")

            rows.append(
                {
                    "Target_Postcode": target,
                    "Shortfall_Leads": shortfall,
                    "Generator_Postcode": generator,
                    "Transfer_Rate": float(c.Transfer_Rate),
                    "eCPR": ecpr,
                    "Budget_Allocation": spend,
                    "Expected_Referrals": expected,
                    "Builder_Recommendations": recs,
                }
            )

            need -= expected
            allocated_for_target += expected
            remaining_budget -= spend
            capacity_remaining[generator] = cap_left - expected

        if allocated_for_target <= 0:
            uncovered.append(target)

    plan_df = pd.DataFrame(rows)
    if plan_df.empty:
        empty["summary"]["postcodes_uncoverable"] = uncovered or shortfalls["Postcode"].astype(str).tolist()
        return empty

    cluster_plan_df = plan_df.copy()
    if "ClusterId" in summary.columns:
        c_map = summary.set_index("Postcode")["ClusterId"].to_dict()
        cluster_plan_df["Target_Cluster"] = cluster_plan_df["Target_Postcode"].map(c_map).fillna(-1).astype(int)
        cluster_plan_df["Generator_Cluster"] = cluster_plan_df["Generator_Postcode"].map(c_map).fillna(-1).astype(int)
        cluster_plan_df = (
            cluster_plan_df.groupby(["Target_Cluster", "Generator_Cluster"], as_index=False)
            .agg(
                Budget_Allocation=("Budget_Allocation", "sum"),
                Expected_Referrals=("Expected_Referrals", "sum"),
                Avg_eCPR=("eCPR", "mean"),
            )
            .sort_values("Budget_Allocation", ascending=False)
        )
    else:
        cluster_plan_df = (
            cluster_plan_df.groupby(["Target_Postcode", "Generator_Postcode"], as_index=False)
            .agg(
                Budget_Allocation=("Budget_Allocation", "sum"),
                Expected_Referrals=("Expected_Referrals", "sum"),
                Avg_eCPR=("eCPR", "mean"),
            )
            .sort_values("Budget_Allocation", ascending=False)
        )

    util = (
        plan_df.groupby("Generator_Postcode", as_index=False)
        .agg(
            Total_Allocated=("Budget_Allocation", "sum"),
            Expected_Referrals=("Expected_Referrals", "sum"),
            Destinations_Served=("Target_Postcode", "nunique"),
        )
        .sort_values("Total_Allocated", ascending=False)
    )
    util["Capacity_Used_Pct"] = (
        util["Expected_Referrals"] / util["Generator_Postcode"].map(capacity).replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100.0
    util = util[["Generator_Postcode", "Total_Allocated", "Capacity_Used_Pct", "Destinations_Served"]]

    total_alloc = float(plan_df["Budget_Allocation"].sum())
    total_refs = float(plan_df["Expected_Referrals"].sum())
    total_shortfall = float(shortfalls["Shortfall"].sum())
    postcodes_covered = int(plan_df["Target_Postcode"].nunique())
    blended_ecpr = total_alloc / total_refs if total_refs > 0 else 0.0

    summary_payload = {
        "total_budget_allocated": total_alloc,
        "total_expected_referrals": total_refs,
        "coverage_pct": (total_refs / total_shortfall) if total_shortfall > 0 else 1.0,
        "blended_ecpr": blended_ecpr,
        "postcodes_covered": postcodes_covered,
        "postcodes_uncoverable": sorted(set(uncovered)),
    }

    return {
        "plan_df": plan_df.sort_values(["Target_Postcode", "eCPR"], ascending=[True, True]).reset_index(drop=True),
        "cluster_plan_df": cluster_plan_df.reset_index(drop=True),
        "summary": summary_payload,
        "generator_utilization": util.reset_index(drop=True),
    }


def compute_postcode_network_metrics(graph: nx.DiGraph, postcode_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add network centrality metrics to postcode summary.
    """
    summary = postcode_summary.copy() if postcode_summary is not None else pd.DataFrame()
    if summary.empty:
        return summary

    for col in ["PageRank", "Betweenness", "InDegree_Weighted", "OutDegree_Weighted"]:
        if col not in summary.columns:
            summary[col] = 0.0

    if graph is None or graph.number_of_nodes() == 0:
        return summary

    pr = nx.pagerank(graph, weight="weight") if graph.number_of_edges() > 0 else {n: 0.0 for n in graph.nodes()}
    btw = nx.betweenness_centrality(graph, weight="weight", normalized=True) if graph.number_of_edges() > 0 else {n: 0.0 for n in graph.nodes()}
    indeg = dict(graph.in_degree(weight="weight"))
    outdeg = dict(graph.out_degree(weight="weight"))

    summary["PageRank"] = summary["Postcode"].map(pr).fillna(0.0)
    summary["Betweenness"] = summary["Postcode"].map(btw).fillna(0.0)
    summary["InDegree_Weighted"] = summary["Postcode"].map(indeg).fillna(0.0)
    summary["OutDegree_Weighted"] = summary["Postcode"].map(outdeg).fillna(0.0)
    return summary


def get_postcode_detail(postcode: str, events_df: pd.DataFrame, mapping_df: pd.DataFrame) -> dict:
    """
    Drill-down payload for a single postcode.
    """
    payload = {
        "postcode": postcode,
        "builders": pd.DataFrame(),
        "monthly_trend": pd.DataFrame(),
        "top_sources": pd.DataFrame(),
        "top_destinations": pd.DataFrame(),
        "spend_trend": pd.DataFrame(),
    }
    if events_df is None or events_df.empty or postcode is None:
        return payload

    df, dest_col, _, _ = _attach_geo_columns(events_df)
    payer_col = _find_col(df.columns, ["MediaPayer_BuilderRegionKey", "_attributed_payer"])
    cost_col = _find_col(df.columns, ["MediaCost_referral_event", "MediaCost_builder_touch", "MediaCost_origin_lead", "MediaCost"])
    lead_date_col = _find_col(df.columns, ["lead_date", "LeadDate", "CreatedDate"])
    ref_date_col = _find_col(df.columns, ["RefDate", "ref_date", "QualifiedDate"])

    df, _, _, _ = prepare_referral_ids(df, inplace=False)
    builder_to_postcode = _flow_builder_to_postcode_map(mapping_df)
    df["_dest_postcode"] = df[dest_col].map(builder_to_postcode).fillna(_derive_region_from_builder_key(df[dest_col]))
    if payer_col:
        df["_payer_postcode"] = df[payer_col].map(builder_to_postcode).fillna(_derive_region_from_builder_key(df[payer_col]))
    else:
        df["_payer_postcode"] = np.nan

    local = df[df["_dest_postcode"].astype(str) == str(postcode)].copy()
    if local.empty:
        return payload

    def _builder_stats(g: pd.DataFrame) -> pd.Series:
        leads, refs, _, _ = count_leads_refs(g)
        spend = pd.to_numeric(g[cost_col], errors="coerce").sum() if cost_col else 0.0
        return pd.Series({"Lead_Count": leads, "Referral_Count": refs, "Media_Spend": float(spend)})

    builders = (
        local.groupby(dest_col, dropna=False)
        .apply(_builder_stats)
        .reset_index()
        .rename(columns={dest_col: "BuilderRegionKey"})
        .sort_values("Lead_Count", ascending=False)
    )
    payload["builders"] = builders

    local["_event_date"] = pd.to_datetime(local[lead_date_col], errors="coerce") if lead_date_col else pd.NaT
    if ref_date_col:
        local["_event_date"] = local["_event_date"].fillna(pd.to_datetime(local[ref_date_col], errors="coerce"))
    local = local[local["_event_date"].notna()].copy()

    if not local.empty:
        local["_month"] = local["_event_date"].dt.to_period("M").dt.start_time

        def _month_stats(g: pd.DataFrame) -> pd.Series:
            leads, refs, _, _ = count_leads_refs(g)
            spend = pd.to_numeric(g[cost_col], errors="coerce").sum() if cost_col else 0.0
            return pd.Series({"Leads": leads, "Referrals": refs, "Media_Spend": float(spend)})

        monthly = local.groupby("_month").apply(_month_stats).reset_index().rename(columns={"_month": "Month"})
        payload["monthly_trend"] = monthly.sort_values("Month")
        payload["spend_trend"] = monthly[["Month", "Media_Spend"]].copy()

    ref_payload = build_postcode_referral_flows(events_df, mapping_df)
    flows = ref_payload.get("flows_df", pd.DataFrame())
    if flows is not None and not flows.empty:
        top_sources = (
            flows[flows["Dest_Postcode"].astype(str) == str(postcode)]
            .sort_values("Referral_Count", ascending=False)
            .head(10)
            .rename(columns={"Origin_Postcode": "Source_Postcode"})
        )
        top_dests = (
            flows[flows["Origin_Postcode"].astype(str) == str(postcode)]
            .sort_values("Referral_Count", ascending=False)
            .head(10)
            .rename(columns={"Dest_Postcode": "Dest_Postcode"})
        )
        payload["top_sources"] = top_sources[["Source_Postcode", "Referral_Count", "Media_Spend"]]
        payload["top_destinations"] = top_dests[["Dest_Postcode", "Referral_Count", "Media_Spend"]]

    return payload
