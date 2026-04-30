import math
import numpy as np
import pandas as pd
import streamlit as st


def get_snowflake_session():
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner="Loading data...")
def load_data(query, _session, start_date: str, end_date: str, country: str):
    safe_country = country.replace("'", "''")
    query = query.format(
        start_date=start_date,
        end_date=end_date,
        country=safe_country,
    )
    df = _session.sql(query).to_pandas()
    df.columns = [c.upper() for c in df.columns]
    return df


def compute_baseline(df, providers):
    baseline = {}
    for prov in providers:
        mask = df["PROVIDER"] == prov
        booked_mask = mask & (df["BOOKED"] == 1)
        baseline[prov] = {
            "clicks": int(mask.sum()),
            "bookings": int(booked_mask.sum()),
            "commission": round(float(df.loc[booked_mask, "TOTALCOMMISSION"].sum()), 2),
        }
    return baseline


def run_simulation(
    df, providers, prob_cols, source_providers, target_provider,
    target_bookings, conversion_rates, commission_pct_change,
):
    source_cols = [f"{sp.upper()}_ADJ_PROB" for sp in source_providers]
    target_col = f"{target_provider.upper()}_ADJ_PROB"

    for col in prob_cols + ["TOTALCOMMISSION"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    total_rows = len(df)
    baseline = compute_baseline(df, providers)

    factor = 1 + commission_pct_change / 100
    tgt_booked = (df["PROVIDER"] == target_provider) & (df["BOOKED"] == 1)
    tgt_avg_comm = float(df.loc[tgt_booked, "TOTALCOMMISSION"].mean()) if tgt_booked.any() else 0.0
    tgt_avg_comm_adj = round(tgt_avg_comm * factor, 2)

    existing_bookings = baseline[target_provider]["bookings"]
    additional_needed = target_bookings - existing_bookings

    source_max = df[source_cols].max(axis=1)

    is_source_top = pd.Series(True, index=df.index)
    for pc in prob_cols:
        is_source_top &= source_max >= df[pc]

    candidates = df[is_source_top & (df[target_col] > 0)].copy()
    candidate_count = len(candidates)

    if candidate_count == 0 or additional_needed <= 0:
        return pd.DataFrame(), baseline, total_rows, candidate_count, tgt_avg_comm_adj

    cand_source_max = source_max[candidates.index].values
    min_boost = (cand_source_max / candidates[target_col].values - 1) * 100

    sort_idx = np.argsort(min_boost)
    sorted_boosts = min_boost[sort_idx]
    sorted_booked = candidates["BOOKED"].values[sort_idx]
    sorted_commission = candidates["TOTALCOMMISSION"].values[sort_idx]
    sorted_provider_names = candidates["PROVIDER"].values[sort_idx]
    cumsum_booked = np.cumsum(sorted_booked)
    cumsum_commission = np.cumsum(sorted_commission)

    results = []
    for cr in conversion_rates:
        switches_needed = math.ceil(additional_needed / cr)

        if switches_needed > len(sorted_boosts):
            results.append({
                "Conversion Rate": cr,
                "Feasible": False,
                "Required Boost %": None,
                "Clicks Switched": switches_needed,
                "% Traffic": round(switches_needed / total_rows * 100, 2) if total_rows else 0,
            })
            continue

        boost = sorted_boosts[switches_needed - 1]
        lost_txns = int(cumsum_booked[switches_needed - 1])
        lost_comm = float(cumsum_commission[switches_needed - 1])
        new_bkgs = round(switches_needed * cr, 1)
        new_comm = round(new_bkgs * tgt_avg_comm_adj, 2)

        row_data = {
            "Conversion Rate": cr,
            "Feasible": True,
            "Required Boost %": round(boost, 2),
            "Clicks Switched": switches_needed,
            "% Traffic": round(switches_needed / total_rows * 100, 2),
            "Sources Lost Txns": lost_txns,
            "Sources Lost Commission": round(lost_comm, 2),
            f"{target_provider} Clicks After": baseline[target_provider]["clicks"] + switches_needed,
            f"{target_provider} Bookings After": round(existing_bookings + new_bkgs, 1),
            f"{target_provider} Commission After": round(baseline[target_provider]["commission"] + new_comm, 2),
        }

        switched_provs = sorted_provider_names[:switches_needed]
        switched_booked = sorted_booked[:switches_needed]
        switched_comm = sorted_commission[:switches_needed]
        for sp in source_providers:
            sp_mask = switched_provs == sp
            sp_switched = int(sp_mask.sum())
            sp_lost_txns = int(switched_booked[sp_mask].sum())
            sp_lost_comm = float(switched_comm[sp_mask].sum())
            row_data[f"{sp} Clicks After"] = baseline[sp]["clicks"] - sp_switched
            row_data[f"{sp} Bookings After"] = baseline[sp]["bookings"] - sp_lost_txns
            row_data[f"{sp} Commission After"] = round(baseline[sp]["commission"] - sp_lost_comm, 2)

        results.append(row_data)

    results_df = pd.DataFrame(results)
    return results_df, baseline, total_rows, candidate_count, tgt_avg_comm_adj
