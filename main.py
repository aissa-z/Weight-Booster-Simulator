import datetime
import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROVIDERS = ["booking", "expedia", "hotelscom"]
PROB_COLS = ["BOOKING_ADJ_PROB", "EXPEDIA_ADJ_PROB", "HOTELSCOM_ADJ_PROB"]

DATA_QUERY = """
SELECT
  c.timestamp,
  c.user_country,
  c.dest_country,
  c.product,
  c.medium,
  c.provider,
  IFF(b._id IS NOT NULL, 1, 0) AS booked,
  totalcommission,
  PARSE_JSON(c.ml)[0]:adjustedProbability::FLOAT AS top1_adj_prob,
  PARSE_JSON(c.ml)[1]:adjustedProbability::FLOAT AS top2_adj_prob,
  PARSE_JSON(c.ml)[2]:adjustedProbability::FLOAT AS top3_adj_prob,
  CASE WHEN PARSE_JSON(c.ml)[0]:provider::STRING = 'booking' THEN PARSE_JSON(c.ml)[0]:adjustedProbability::FLOAT
       WHEN PARSE_JSON(c.ml)[1]:provider::STRING = 'booking' THEN PARSE_JSON(c.ml)[1]:adjustedProbability::FLOAT
       WHEN PARSE_JSON(c.ml)[2]:provider::STRING = 'booking' THEN PARSE_JSON(c.ml)[2]:adjustedProbability::FLOAT
  END AS booking_adj_prob,
  CASE WHEN PARSE_JSON(c.ml)[0]:provider::STRING = 'expedia' THEN PARSE_JSON(c.ml)[0]:adjustedProbability::FLOAT
       WHEN PARSE_JSON(c.ml)[1]:provider::STRING = 'expedia' THEN PARSE_JSON(c.ml)[1]:adjustedProbability::FLOAT
       WHEN PARSE_JSON(c.ml)[2]:provider::STRING = 'expedia' THEN PARSE_JSON(c.ml)[2]:adjustedProbability::FLOAT
  END AS expedia_adj_prob,
  CASE WHEN PARSE_JSON(c.ml)[0]:provider::STRING = 'hotelscom' THEN PARSE_JSON(c.ml)[0]:adjustedProbability::FLOAT
       WHEN PARSE_JSON(c.ml)[1]:provider::STRING = 'hotelscom' THEN PARSE_JSON(c.ml)[1]:adjustedProbability::FLOAT
       WHEN PARSE_JSON(c.ml)[2]:provider::STRING = 'hotelscom' THEN PARSE_JSON(c.ml)[2]:adjustedProbability::FLOAT
  END AS hotelscom_adj_prob
FROM product.transformed.events_pubsub_clicks c
LEFT JOIN ENGINEERING.TRANSFORMED.MONGO_HUB_RELEASE_BOOKINGS b
  ON b.click_id = c.click_id
WHERE c.category = 'accommodation'
  AND c.is_wildcarded = TRUE
  AND c.is_biased = FALSE
  AND c.provider <> 'airbnb'
  AND c.timestamp BETWEEN '{start_date}' AND '{end_date}'
  AND c.user_country = '{country}'
"""


# ---------------------------------------------------------------------------
# Snowflake connection
# ---------------------------------------------------------------------------

def get_snowflake_session():
    try:
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner="Loading data...")
def load_data(_session, start_date: str, end_date: str, country: str):
    """Load filtered click data from Snowflake using the predefined query."""
    safe_country = country.replace("'", "''")
    query = DATA_QUERY.format(
        start_date=start_date,
        end_date=end_date,
        country=safe_country,
    )
    df = _session.sql(query).to_pandas()
    df.columns = [c.upper() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

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
    """
    Run the boost simulation.

    source_providers is a list of providers to take clicks from.
    Returns (results_df, baseline, total_rows, candidate_count, tgt_avg_comm_adj).
    """
    source_cols = [f"{sp.upper()}_ADJ_PROB" for sp in source_providers]
    target_col = f"{target_provider.upper()}_ADJ_PROB"

    # Clean numerics
    for col in prob_cols + ["TOTALCOMMISSION"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    total_rows = len(df)
    baseline = compute_baseline(df, providers)

    # Target provider avg commission with adjustment
    factor = 1 + commission_pct_change / 100
    tgt_booked = (df["PROVIDER"] == target_provider) & (df["BOOKED"] == 1)
    tgt_avg_comm = float(df.loc[tgt_booked, "TOTALCOMMISSION"].mean()) if tgt_booked.any() else 0.0
    tgt_avg_comm_adj = round(tgt_avg_comm * factor, 2)

    existing_bookings = baseline[target_provider]["bookings"]
    additional_needed = target_bookings - existing_bookings

    # Max probability among source providers per row
    source_max = df[source_cols].max(axis=1)

    # Candidates: a source provider is top across ALL providers AND target has nonzero prob
    is_source_top = pd.Series(True, index=df.index)
    for pc in prob_cols:
        is_source_top &= source_max >= df[pc]

    candidates = df[is_source_top & (df[target_col] > 0)].copy()
    candidate_count = len(candidates)

    if candidate_count == 0 or additional_needed <= 0:
        return pd.DataFrame(), baseline, total_rows, candidate_count, tgt_avg_comm_adj

    # Min boost per candidate: target must beat the best source prob
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

        # Per-source provider breakdown
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


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def render_boost_chart(results_df):
    feasible = results_df[results_df["Feasible"]].copy()
    if feasible.empty:
        return

    feasible["CR_Label"] = feasible["Conversion Rate"].astype(str)
    feasible["Boost_Label"] = feasible["Required Boost %"].apply(lambda v: f"{v:.1f}%")

    base = alt.Chart(feasible).encode(
        x=alt.X("CR_Label:N", title="Conversion Rate",
                sort=feasible["CR_Label"].tolist()),
    )

    line = base.mark_line(point=True, color="#1565C0", strokeWidth=2).encode(
        y=alt.Y("Required Boost %:Q", title="Required Boost (%)"),
    )

    labels = base.mark_text(dy=-12, color="#1565C0", fontWeight="bold", fontSize=11).encode(
        y=alt.Y("Required Boost %:Q"),
        text="Boost_Label:N",
    )

    bars = base.mark_bar(color="#FF9800", opacity=0.3).encode(
        y=alt.Y("Clicks Switched:Q", title="Clicks Switched"),
    )

    boost_chart = alt.layer(bars, line, labels).resolve_scale(
        y="independent"
    ).properties(height=300)

    st.altair_chart(boost_chart, use_container_width=True)


def render_commission_chart(results_df, baseline, source_providers, target_provider):
    feasible = results_df[results_df["Feasible"]].copy()
    if feasible.empty:
        return

    src_label = " + ".join(sp.title() for sp in source_providers)
    tgt = target_provider
    lost = feasible["Sources Lost Commission"].values
    gained = feasible[f"{tgt} Commission After"].values - baseline[tgt]["commission"]
    net = gained - lost

    chart_data = pd.DataFrame({
        "Conversion Rate": list(feasible["Conversion Rate"].astype(str)) * 3,
        "Commission ($)": list(lost) + list(gained) + list(net),
        "Category": (
            [f"{src_label} Lost"] * len(lost)
            + [f"{tgt.title()} Gained"] * len(gained)
            + ["Net Impact"] * len(net)
        ),
    })

    chart_data["Label"] = chart_data["Commission ($)"].apply(lambda v: f"${v:+,.0f}")
    cr_order = feasible["Conversion Rate"].astype(str).tolist()
    cat_order = [f"{src_label} Lost", f"{tgt.title()} Gained", "Net Impact"]

    bars = alt.Chart(chart_data).mark_bar(opacity=0.8).encode(
        x=alt.X("Conversion Rate:N", title="Conversion Rate", sort=cr_order),
        y=alt.Y("Commission ($):Q", title="Commission ($)"),
        color=alt.Color("Category:N", sort=cat_order,
                        scale=alt.Scale(
                            domain=cat_order,
                            range=["#C62828", "#2E7D32", "#1565C0"],
                        )),
        xOffset=alt.XOffset("Category:N", sort=cat_order),
    )

    text = alt.Chart(chart_data).mark_text(dy=-10, fontSize=9, fontWeight="bold").encode(
        x=alt.X("Conversion Rate:N", sort=cr_order),
        y=alt.Y("Commission ($):Q"),
        text="Label:N",
        color=alt.Color("Category:N", sort=cat_order,
                        scale=alt.Scale(
                            domain=cat_order,
                            range=["#C62828", "#2E7D32", "#1565C0"],
                        ), legend=None),
        xOffset=alt.XOffset("Category:N", sort=cat_order),
    )

    commission_chart = (bars + text).properties(height=300)
    st.altair_chart(commission_chart, use_container_width=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Provider Boost Simulation", layout="wide")
    st.title("Provider Probability Boost Simulation")

    st.info(
        "**Disclaimer:** Only clicks resulting in a booking within the selected period are taken into account. "
        "If a booking occurred inside the time window but the click happened outside, it will not be considered. "
        "Only clicks that can be affected by the SO model are included (wildcarded, non-biased, accommodation only). "
        "Airbnb is not taken into account as it uses a separate model."
    )

    session = get_snowflake_session()
    if session is None:
        st.error("Could not connect to Snowflake. Ensure this app is running in Streamlit in Snowflake.")
        st.stop()

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Configuration")

        # Filters
        st.subheader("Filters")
        country = st.text_input("Country code", value="GB",
                                help="ISO 2-letter country code (e.g. GB, US, FR)")

        today = datetime.date.today()
        default_start = today - datetime.timedelta(days=30)
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input("Start date", value=default_start)
        with date_col2:
            end_date = st.date_input("End date", value=today)

        if start_date > end_date:
            st.error("Start date must be before end date.")
            st.stop()

        providers = PROVIDERS
        prob_cols = PROB_COLS

        # Simulation parameters
        st.subheader("Simulation Parameters")
        source_providers = st.multiselect(
            "Source providers (take clicks from)",
            providers,
            default=["booking"],
        )
        remaining = [p for p in providers if p not in source_providers]
        target_providers = st.multiselect(
            "Target providers (boost)",
            remaining,
            default=[p for p in ["hotelscom"] if p in remaining],
        )

        if not source_providers or not target_providers:
            st.warning("Select at least one source and one target provider.")
            st.stop()

        target_bookings = st.number_input("Target bookings", min_value=1, value=1500, step=100)

        cr_input = st.text_input(
            "Conversion rates (comma-separated)",
            value="0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003",
        )
        try:
            conversion_rates = sorted([float(x.strip()) for x in cr_input.split(",") if x.strip()])
        except ValueError:
            st.error("Invalid conversion rates. Use comma-separated decimals.")
            st.stop()

        commission_pct = st.number_input(
            "Commission % adjustment",
            value=17.65,
            step=0.5,
            format="%.2f",
            help="Percentage change applied to the target provider's historical avg commission. E.g. 17.65 means x1.1765.",
        )

        run = st.button("Run Simulation", type="primary", use_container_width=True)

    # ---- Main area ----
    if not run:
        st.info("Configure parameters in the sidebar and click **Run Simulation**.")
        st.stop()

    # Load data
    df = load_data(session, str(start_date), str(end_date), country)

    if df.empty:
        st.warning(f"No data found for country **{country}** between {start_date} and {end_date}.")
        st.stop()

    src_label = " + ".join(sp.title() for sp in source_providers)

    # ---- Baseline (shared across all targets) ----
    # Clean numerics once
    for col in prob_cols + ["TOTALCOMMISSION"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    baseline = compute_baseline(df, providers)

    st.markdown("---")
    st.subheader("Baseline")
    baseline_data = []
    for prov in providers:
        b = baseline[prov]
        baseline_data.append({
            "Provider": prov.title(),
            "Clicks": f"{b['clicks']:,}",
            "Bookings": f"{b['bookings']:,}",
            "Commission ($)": f"${b['commission']:,.2f}",
        })
    st.dataframe(pd.DataFrame(baseline_data), hide_index=True, use_container_width=True)

    # ---- Per-target simulation in tabs ----
    tabs = st.tabs([tp.title() for tp in target_providers])

    for tab, target_provider in zip(tabs, target_providers):
        with tab:
            result = run_simulation(
                df, providers, prob_cols, source_providers, target_provider,
                target_bookings, conversion_rates, commission_pct,
            )
            results_df, _, total_rows, candidate_count, tgt_avg_comm = result

            # KPIs
            st.markdown("---")
            kpi_cols = st.columns(5)
            kpi_cols[0].metric("Total Clicks", f"{total_rows:,}")
            kpi_cols[1].metric("Switchable Candidates", f"{candidate_count:,}")
            kpi_cols[2].metric(f"Current {target_provider.title()} Bookings", f"{baseline[target_provider]['bookings']:,}")
            kpi_cols[3].metric("Target Bookings", f"{target_bookings:,}")
            kpi_cols[4].metric(f"{target_provider.title()} Avg Comm (adj.)", f"${tgt_avg_comm:.2f}")

            if results_df.empty:
                if baseline[target_provider]["bookings"] >= target_bookings:
                    st.success(f"{target_provider.title()} already has {baseline[target_provider]['bookings']:,} bookings (target: {target_bookings:,}).")
                else:
                    st.warning("No switchable candidates found.")
                continue

            # Scenario Results
            st.markdown("---")
            st.subheader("Scenario Results")

            feasible = results_df[results_df["Feasible"]].copy()
            infeasible = results_df[~results_df["Feasible"]]

            if not infeasible.empty:
                st.warning(
                    f"{len(infeasible)} scenario(s) are infeasible — not enough candidate clicks "
                    f"at those conversion rates to reach {target_bookings:,} bookings."
                )

            if feasible.empty:
                st.error("No feasible scenarios. Try lowering the target or increasing conversion rates.")
                continue

            # Build display table
            src_lost_txns_col = f"{src_label} Lost Txns"
            src_lost_col = f"{src_label} Lost $"
            tgt_gained_col = f"{target_provider.title()} Gained $"

            display_rows = []
            for _, r in feasible.iterrows():
                gained_comm = r[f"{target_provider} Commission After"] - baseline[target_provider]["commission"]
                lost_comm = r["Sources Lost Commission"]
                net = gained_comm - lost_comm
                display_rows.append({
                    "Conv. Rate": r["Conversion Rate"],
                    "Boost %": f"{r['Required Boost %']:.1f}%",
                    "Clicks Switched": f"{int(r['Clicks Switched']):,}",
                    "% Traffic": f"{r['% Traffic']:.1f}%",
                    src_lost_txns_col: f"{int(r['Sources Lost Txns']):,}",
                    src_lost_col: f"-${lost_comm:,.0f}",
                    tgt_gained_col: f"+${gained_comm:,.0f}",
                    "Net Impact $": f"${net:+,.0f}",
                })

            scenario_col_config = {
                "Conv. Rate": st.column_config.TextColumn(
                    "Conv. Rate",
                    help="Assumed conversion rate: fraction of switched clicks that result in a booking for the target provider.",
                ),
                "Boost %": st.column_config.TextColumn(
                    "Boost %",
                    help="Minimum multiplicative boost (%) applied to the target provider's probability so it surpasses all source providers.",
                ),
                "Clicks Switched": st.column_config.TextColumn(
                    "Clicks Switched",
                    help="Number of clicks redirected from source providers to the target provider at this conversion rate.",
                ),
                "% Traffic": st.column_config.TextColumn(
                    "% Traffic",
                    help="Percentage of total clicks that would be switched to the target provider.",
                ),
                src_lost_txns_col: st.column_config.TextColumn(
                    src_lost_txns_col,
                    help="Number of bookings (BOOKED=1) lost by source providers due to their clicks being redirected.",
                ),
                src_lost_col: st.column_config.TextColumn(
                    src_lost_col,
                    help="Total commission lost by source providers from the redirected clicks that had a booking.",
                ),
                tgt_gained_col: st.column_config.TextColumn(
                    tgt_gained_col,
                    help="Estimated commission gained by the target provider from new bookings, calculated as (Target bookings - Current bookings) x avg. commission x (1 + commission boost %).",
                ),
                "Net Impact $": st.column_config.TextColumn(
                    "Net Impact $",
                    help="Net commission change: target gained minus sources lost. Positive means overall commission increase.",
                ),
            }

            st.dataframe(
                pd.DataFrame(display_rows),
                hide_index=True,
                use_container_width=True,
                column_config=scenario_col_config,
            )

            # Charts
            st.markdown("---")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.subheader("Boost % vs Conversion Rate")
                render_boost_chart(results_df)

            with chart_col2:
                st.subheader("Commission Impact ($)")
                render_commission_chart(results_df, baseline, source_providers, target_provider)

            # Provider impact detail (best-case)
            st.markdown("---")
            best = feasible.iloc[-1]
            st.subheader(f"Provider Impact Detail (Best Case: CR = {best['Conversion Rate']})")

            impact_rows = []
            for prov in providers:
                before_c = baseline[prov]["clicks"]
                before_b = baseline[prov]["bookings"]
                before_cm = baseline[prov]["commission"]

                if prov in source_providers and f"{prov} Clicks After" in best.index:
                    after_c = int(best[f"{prov} Clicks After"])
                    after_b = int(best[f"{prov} Bookings After"])
                    after_cm = best[f"{prov} Commission After"]
                elif prov == target_provider:
                    after_c = int(best[f"{target_provider} Clicks After"])
                    after_b = best[f"{target_provider} Bookings After"]
                    after_cm = best[f"{target_provider} Commission After"]
                else:
                    after_c = before_c
                    after_b = before_b
                    after_cm = before_cm

                delta_cm = after_cm - before_cm
                impact_rows.append({
                    "Provider": prov.title(),
                    "Clicks Before": f"{before_c:,}",
                    "Clicks After": f"{after_c:,}",
                    "Bookings Before": f"{before_b:,}",
                    "Bookings After": f"{int(after_b) if isinstance(after_b, (int, np.integer)) else after_b:,}",
                    "Commission Before": f"${before_cm:,.2f}",
                    "Commission After": f"${after_cm:,.2f}",
                    "Delta Commission": f"${delta_cm:+,.2f}",
                })

            st.dataframe(pd.DataFrame(impact_rows), hide_index=True, use_container_width=True)

            st.caption(
                f"**Methodology:** Multiplicative boost applied to {target_provider.title()}'s probability to surpass "
                f"{src_label}. Only {src_label} recommendations are switched; other providers are unchanged. "
                f"New {target_provider.title()} bookings estimated as switched clicks x conversion rate. "
                f"Commission uses historical avg (${tgt_avg_comm:.2f}/booking, adjusted by {commission_pct:+.2f}%)."
            )


if __name__ == "__main__":
    main()