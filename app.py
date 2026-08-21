"""
Transaction Anomaly Explainer — Interactive Risk Analyst Triage Dashboard.
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic import generate_synthetic_transactions
from src.pipeline import TransactionAnomalyPipeline

EVAL_RESULTS_PATH = PROJECT_ROOT / "evaluation_results.json"

st.set_page_config(
    page_title="Transaction Anomaly Explainer | Razorpay Buildathon",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        color: #f8fafc;
    }
    .badge-flagged {
        background-color: #ef4444;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-clean {
        background-color: #10b981;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-grounded {
        background-color: #0284c7;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .explanation-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        border-radius: 4px;
        padding: 14px;
        font-size: 1.05rem;
        line-height: 1.5;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    return TransactionAnomalyPipeline()


def _load_eval_metrics() -> dict:
    if EVAL_RESULTS_PATH.exists():
        with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_sample_queue():
    pipeline = load_pipeline()
    df = generate_synthetic_transactions(normal_count=40, anomaly_count=15)
    results = pipeline.analyze_synthetic_batch(df)
    return results


def main():
    pipeline = load_pipeline()

    st.sidebar.image("https://razorpay.com/assets/razorpay-logo.svg", width=180)
    st.sidebar.title("AI Risk Manager")
    st.sidebar.markdown("**Transaction Anomaly Explainer**")
    st.sidebar.caption("Track: AI Risk Manager | Razorpay AI Buildathon")
    st.sidebar.divider()

    eval_data = _load_eval_metrics()
    det = eval_data.get("detection_metrics", {})
    exp = eval_data.get("explanation_metrics", {})

    st.sidebar.subheader("Held-Out Eval Benchmarks")
    if det:
        st.sidebar.markdown(
            f"- **Detection Precision:** `{det.get('precision', 'N/A')}`\n"
            f"- **Detection Recall:** `{det.get('recall', 'N/A')}`\n"
            f"- **Detection F1 Score:** `{det.get('f1', 'N/A')}`\n"
            f"- **Decision Threshold:** `{det.get('threshold', 'N/A')}` (tuned on val set)\n"
            f"- **Explanation Accuracy:** `{exp.get('explanation_accuracy_pct', 'N/A')}%`\n"
            f"- **Grounding Pass Rate:** `{exp.get('grounding_compliance_pct', 'N/A')}%`"
        )
        st.sidebar.caption(
            f"Evaluated on Kaggle test set ({det.get('test_rows', '?'):,} rows) "
            f"& {exp.get('sample_size', 20)}-sample hand-graded rubric."
        )
    else:
        st.sidebar.warning(
            "Eval results not found. Run `python scripts/run_evaluation.py` to generate them."
        )

    if not pipeline.classifier_ready:
        st.sidebar.error(
            "XGBoost classifier not loaded.\n\n"
            "Kaggle-path analysis is unavailable.\n\n"
            "Run: `python scripts/train_classifier.py`"
        )

    st.sidebar.divider()

    st.sidebar.subheader("System Architecture")
    st.sidebar.info(
        "**Two-Layer Invariant:**\n"
        "1. Detection Layer evaluates quantitative rules & XGBoost.\n"
        "2. Explanation Layer receives **ONLY** `triggered_signals`."
    )

    tab_pitch, tab_queue, tab_simulator = st.tabs([
        "Pitch Demo Scenarios",
        "Live Risk Operations Queue",
        "Interactive Transaction Simulator",
    ])

    with tab_pitch:
        st.header("5-Minute Pitch Demo Scenarios")
        st.markdown(
            "Curated scenarios demonstrating clean catches, false-positive transparency, "
            "and low-confidence handling."
        )

        scenario_choice = st.radio(
            "Select Pitch Scenario:",
            [
                "1. Clean Catch (High-Confidence Blatant Anomaly)",
                "2. False-Positive Walkthrough (Explaining 'Why' to Avoid Alert Fatigue)",
                "3. Low-Confidence Edge Case (System Uncertainty Expression)",
            ],
            horizontal=True,
        )

        if "1. Clean Catch" in scenario_choice:
            st.subheader("Scenario 1: Clean Catch — Blatant Anomaly")
            st.markdown(
                "**Narrative:** A compromised card where an attacker attempts a massive off-hours purchase "
                "from an unrecognized device abroad."
            )
            clean_catch_txn = {
                "transaction_id": "txn_demo_clean_catch",
                "user_id": "usr_0042",
                "amount": 28500.00,
                "historical_avg_amount": 2500.00,
                "location_lat": 51.5074,
                "location_lon": -0.1278,
                "historical_location_lat": 12.9716,
                "historical_location_lon": 77.5946,
                "device_id": "dev_unrecognized_99812",
                "historical_device_ids": json.dumps(["dev_usr_0042_1"]),
                "timestamp": "2026-08-20T03:15:00",
                "typical_active_start_hour": 8,
                "typical_active_end_hour": 22,
            }
            result = pipeline.analyze_synthetic_transaction(clean_catch_txn)
            render_audit_card(result, clean_catch_txn)

        elif "2. False-Positive" in scenario_choice:
            st.subheader("Scenario 2: False Positive Walkthrough — Transparent Reasoning")
            st.markdown(
                "**Narrative:** A legitimate VIP customer traveling to New York makes an expensive electronics "
                "purchase at 2 AM. The system flags the anomaly and clearly states the exact reasons, enabling "
                "the analyst to verify travel context and dismiss in seconds."
            )
            fp_txn = {
                "transaction_id": "txn_demo_false_positive",
                "user_id": "usr_0188",
                "amount": 14200.00,
                "historical_avg_amount": 3500.00,
                "location_lat": 40.7128,
                "location_lon": -74.0060,
                "historical_location_lat": 19.0760,
                "historical_location_lon": 72.8777,
                "device_id": "dev_usr_0188_1",
                "historical_device_ids": json.dumps(["dev_usr_0188_1"]),
                "timestamp": "2026-08-20T02:30:00",
                "typical_active_start_hour": 9,
                "typical_active_end_hour": 23,
            }
            result = pipeline.analyze_synthetic_transaction(fp_txn)
            render_audit_card(result, fp_txn, analyst_action="Dismissed (Legitimate Overseas Travel)")

        else:
            st.subheader("Scenario 3: Low-Confidence Edge Case — Honest Uncertainty")
            st.markdown(
                "**Narrative:** A single borderline signal triggered (minor off-hours purchase with modest amount). "
                "Instead of overclaiming high risk, the system transparently reports medium/low confidence."
            )
            edge_txn = {
                "transaction_id": "txn_demo_edge_case",
                "user_id": "usr_0099",
                "amount": 4200.00,
                "historical_avg_amount": 1200.00,
                "location_lat": 12.9716,
                "location_lon": 77.5946,
                "historical_location_lat": 12.9716,
                "historical_location_lon": 77.5946,
                "device_id": "dev_usr_0099_1",
                "historical_device_ids": json.dumps(["dev_usr_0099_1"]),
                "timestamp": "2026-08-20T04:10:00",
                "typical_active_start_hour": 8,
                "typical_active_end_hour": 22,
            }
            result = pipeline.analyze_synthetic_transaction(edge_txn)
            render_audit_card(result, edge_txn)

    with tab_queue:
        st.header("Risk Operations Queue")
        st.markdown("Triage queue for operations analysts to review flagged items.")

        queue_data = load_sample_queue()

        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            filter_mode = st.selectbox("Queue Filter", ["All Transactions", "Flagged Anomalies Only", "Clean Transactions Only"])

        filtered = queue_data
        if filter_mode == "Flagged Anomalies Only":
            filtered = [r for r in queue_data if r["flag"]]
        elif filter_mode == "Clean Transactions Only":
            filtered = [r for r in queue_data if not r["flag"]]

        table_rows = []
        for r in filtered:
            table_rows.append({
                "Txn ID": r["transaction_id"],
                "Status": "FLAGGED" if r["flag"] else "CLEAN",
                "Risk Score": f"{r['risk_score']:.2f}",
                "Confidence": r["confidence"].upper(),
                "Signals Count": len(r["triggered_signals"]),
                "Primary Signals": ", ".join(r["triggered_signals"]) if r["triggered_signals"] else "None",
            })

        df_display = pd.DataFrame(table_rows)
        st.dataframe(df_display, use_container_width=True, height=260)

        st.divider()
        st.subheader("Transaction Detail Inspector")
        selected_id = st.selectbox("Select Transaction ID to inspect:", [r["transaction_id"] for r in filtered])

        if selected_id:
            selected_item = next(r for r in filtered if r["transaction_id"] == selected_id)
            render_audit_card(selected_item)

    with tab_simulator:
        st.header("Custom Transaction Simulator")
        st.markdown("Simulate any transaction to test detection and explanation generation in real-time.")

        c1, c2, c3 = st.columns(3)
        with c1:
            sim_amount = st.number_input("Transaction Amount (Rs / $)", min_value=10.0, max_value=500000.0, value=18500.0, step=500.0)
            sim_avg_amount = st.number_input("User Historical Avg Amount", min_value=10.0, max_value=50000.0, value=2500.0, step=200.0)

        with c2:
            sim_dist_km = st.slider("Distance from User Usual Location (km)", min_value=0, max_value=15000, value=1200, step=50)
            sim_velocity = st.slider("2-Minute Transaction Count (Velocity)", min_value=1, max_value=10, value=1)

        with c3:
            sim_new_dev = st.checkbox("Unrecognized Device?", value=True)
            sim_odd_hour = st.checkbox("Off-Hours / Night Transaction?", value=False)

        if st.button("Analyze Simulated Transaction", type="primary"):
            sim_txn = {
                "transaction_id": "txn_sim_custom",
                "user_id": "usr_sim_01",
                "amount": float(sim_amount),
                "historical_avg_amount": float(sim_avg_amount),
                "location_lat": 12.9716 + (sim_dist_km / 111.0),
                "location_lon": 77.5946,
                "historical_location_lat": 12.9716,
                "historical_location_lon": 77.5946,
                "device_id": "dev_new_sim" if sim_new_dev else "dev_known",
                "historical_device_ids": json.dumps(["dev_known"]),
                "timestamp": "2026-08-20T03:00:00" if sim_odd_hour else "2026-08-20T14:00:00",
                "typical_active_start_hour": 8,
                "typical_active_end_hour": 22,
            }
            sim_history = []
            if sim_velocity > 1:
                for i in range(sim_velocity - 1):
                    sim_history.append({
                        "user_id": "usr_sim_01",
                        "timestamp": "2026-08-20T13:59:30",
                    })

            res = pipeline.analyze_synthetic_transaction(sim_txn, recent_history=sim_history)
            render_audit_card(res, sim_txn)


def render_audit_card(record: dict, raw_txn: dict | None = None, analyst_action: str | None = None):
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 2, 2])

    with col1:
        st.markdown(f"**Transaction ID:** `{record['transaction_id']}`")
        if record["flag"]:
            st.markdown('<span class="badge-flagged">ANOMALY FLAGGED</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-clean">CLEAN / NORMAL</span>', unsafe_allow_html=True)

    with col2:
        st.metric("Risk Score", f"{record['risk_score']:.2f}")

    with col3:
        st.markdown(f"**Confidence:** `{record['confidence'].upper()}`")
        st.caption(record["confidence_justification"])

    with col4:
        if record["is_grounded"]:
            st.markdown('<span class="badge-grounded">Grounded (Audited)</span>', unsafe_allow_html=True)
            st.caption("Zero hallucinated signals detected.")
        else:
            st.error("Grounding Audit Failed")

    st.markdown("#### Natural Language Grounded Explanation")
    st.markdown(f'<div class="explanation-box">{record["explanation"]}</div>', unsafe_allow_html=True)

    if analyst_action:
        st.success(f"**Recommended Analyst Action:** {analyst_action}")

    with st.expander("Inspect triggered_signals payload (grounding contract)"):
        st.json(record["triggered_signals_detail"])

    if raw_txn:
        with st.expander("View raw transaction metadata"):
            st.json(raw_txn)


if __name__ == "__main__":
    main()
