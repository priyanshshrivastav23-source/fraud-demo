# dashboard.py
# Hackathon-ready Streamlit dashboard
# Team: NullOps
# Contact: priyanshshrivastav23@gmail.com

import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime

# optional: nicer charts
try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# -------------------------
# Page setup + styling
# -------------------------
st.set_page_config(page_title="NullOps — Fraud Detection", layout="wide", page_icon="💳")

# small CSS for nicer header, blinking alert, and table coloring
st.markdown(
    """
    <style>
    .header-title { font-size:32px; font-weight:700; color:#0b6e4f; }
    .subtle { color: #555; }
    .blink { animation: blinker 1.5s linear infinite; color: #b80f0f; font-weight:700; }
    @keyframes blinker { 50% { opacity: 0.0; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Constants / Globals
# -------------------------
TEAM_NAME = "NullOps"
CONTACT_EMAIL = "priyanshshrivastav23@gmail.com"
COUNTRIES = ["India", "USA", "UK", "Germany", "Canada", "China", "UAE", "Singapore"]
MERCHANTS = ["Amazon", "Flipkart", "Walmart", "Myntra", "Target", "Dominos", "Uber", "Ebay"]
DEVICES = ["Mobile", "Desktop", "Tablet", "POS"]
HOME_COUNTRY = "India"
INR_PER_USD = 83.0

# -------------------------
# Session state initialization
# -------------------------
if "transactions" not in st.session_state:
    st.session_state.transactions = pd.DataFrame(
        columns=["TxnID","UserID","Amount","Currency","Location","Merchant","Device","Hour","DayOfWeek","Score","Status","Reason","Timestamp"]
    )
if "running" not in st.session_state:
    st.session_state.running = False
if "txn_counter" not in st.session_state:
    st.session_state.txn_counter = 1

# -------------------------
# Rule-based demo scorer
# -------------------------
def score_and_label(txn: dict, suspicious_th=0.5, fraud_th=0.8):
    reasons = []
    score = 0.0
    try:
        amt = float(txn.get("Amount", 0.0))
    except Exception:
        amt = 0.0

    if amt >= 5000:
        score += 0.45; reasons.append("Very high amount")
    elif amt >= 2500:
        score += 0.25; reasons.append("High amount")

    if txn.get("Location") and txn["Location"] != HOME_COUNTRY:
        score += 0.30; reasons.append("Foreign location")

    hr = int(txn.get("Hour", 12))
    if hr < 6 or hr > 22:
        score += 0.20; reasons.append("Odd hour")

    if txn.get("Device") in ["POS", "Desktop"]:
        score += 0.10; reasons.append(f"Device={txn['Device']}")

    if txn.get("Merchant") in ["Dominos", "Uber"]:
        score += 0.05; reasons.append(f"Merchant={txn['Merchant']}")

    score = min(score, 1.0)

    if score >= fraud_th:
        status = "Fraud"
    elif score >= suspicious_th:
        status = "Suspicious"
    else:
        status = "Safe"

    if not reasons:
        reasons = ["No obvious anomaly"]

    return score, status, ", ".join(reasons)

# -------------------------
# Fake transaction generator
# -------------------------
def gen_txn(txn_id:int, currency="₹"):
    return {
        "TxnID": f"T{txn_id:06d}",
        "UserID": f"U{random.randint(1,9999):04d}",
        "Amount": round(random.uniform(50, 10000),2),
        "Currency": currency,
        "Location": random.choice(COUNTRIES),
        "Merchant": random.choice(MERCHANTS),
        "Device": random.choice(DEVICES),
        "Hour": random.randint(0,23),
        "DayOfWeek": random.randint(0,6),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# -------------------------
# Utilities
# -------------------------
def currency_fmt(amount_in_inr:float, symbol:str):
    if symbol == "₹":
        return f"₹{amount_in_inr:,.2f}"
    else:
        return f"${amount_in_inr/INR_PER_USD:,.2f}"

def status_emoji(status:str):
    if status == "Fraud":
        return "🔴 Fraud"
    if status == "Suspicious":
        return "🟡 Suspicious"
    return "🟢 Safe"

def color_rows(df_display):
    def row_style(row):
        if row.Status == "Fraud":
            return ["background-color:#ffcccc"]*len(row)
        elif row.Status == "Suspicious":
            return ["background-color:#fff2cc"]*len(row)
        else:
            return ["background-color:#ccffcc"]*len(row)
    return pd.DataFrame([row_style(r) for _, r in df_display.iterrows()], index=df_display.index)

# -------------------------
# Navigation
# -------------------------
st.sidebar.title("NullOps — Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Real-Time Simulation", "Explainability", "About"])

# -------------------------
# HOME page
# -------------------------
if page == "Home":
    st.markdown("<div class='header-title'>NullOps — AI-Powered Real-Time Fraud Detection</div>", unsafe_allow_html=True)
    st.write("### Problem statement")
    st.write("""
    Financial fraud costs companies and users billions annually. Fraud is often subtle (unusual location, odd time, sudden large amounts),
    and platforms need fast, explainable, and low-latency detection that adapts via feedback.
    """)
    st.write("### Our Solution")
    st.write("""
    **NullOps** provides:
    - Real-time transaction monitoring (streaming simulation)
    - Transparent scoring + explainability (why a transaction was flagged)
    - Multi-input support: manual, CSV upload, or live stream
    - Clear KPIs and exportable alerts
    """)
    st.write("### Quick actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Open Dashboard"):
            st.experimental_rerun()
    with col2:
        st.write("Contact:")
        st.write(f"**Team:** {TEAM_NAME}  \n**Email:** {CONTACT_EMAIL}")

# -------------------------
# DASHBOARD page
# -------------------------
elif page == "Dashboard":
    st.header("📊 Dashboard — Input & Visuals")
    st.markdown("Use the controls on the left to provide data: manual input, CSV upload, or go to Real-Time Simulation.")

    left, right = st.columns([1,3])
    with left:
        st.subheader("Input")
        input_mode = st.selectbox("Mode", ["Manual Input", "Upload CSV"], key="input_mode")

        if input_mode == "Manual Input":
            with st.form("manual_txn"):
                st.write("Enter transaction details")
                uid = st.text_input("User ID", value=f"U{random.randint(100,999)}")
                amt = st.number_input("Amount (INR)", min_value=1.0, value=500.0, step=1.0)
                loc = st.selectbox("Location", COUNTRIES)
                merch = st.selectbox("Merchant", MERCHANTS)
                dev = st.selectbox("Device", DEVICES)
                hr = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)
                submitted = st.form_submit_button("Add & Score")
            if submitted:
                txn = {
                    "TxnID": f"M{st.session_state.txn_counter:06d}",
                    "UserID": uid,
                    "Amount": amt,
                    "Currency": "₹",
                    "Location": loc,
                    "Merchant": merch,
                    "Device": dev,
                    "Hour": hr,
                    "DayOfWeek": datetime.now().weekday(),
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                score, status, reason = score_and_label(txn, suspicious_th=0.5, fraud_th=0.8)
                txn.update({"Score": score, "Status": status, "Reason": reason})
                st.session_state.txn_counter += 1
                st.session_state.transactions = pd.concat([st.session_state.transactions, pd.DataFrame([txn])], ignore_index=True)
                st.success(f"Added transaction — Status: {status} | Reason: {reason}")

        else:  # Upload CSV
            uploaded = st.file_uploader("Upload CSV (columns: Amount, Location, Merchant, Device, Hour[, UserID, TxnID])", type=["csv"])
            if uploaded:
                try:
                    df_up = pd.read_csv(uploaded)
                    st.write("Preview (first 5 rows):")
                    st.dataframe(df_up.head(), use_container_width=True)
                    if st.button("Score & Add to Session"):
                        rows = []
                        for idx, r in df_up.iterrows():
                            txn = {
                                "TxnID": r.get("TxnID", f"C{st.session_state.txn_counter:06d}"),
                                "UserID": r.get("UserID", f"U{random.randint(100,999)}"),
                                "Amount": float(r.get("Amount", 0.0)),
                                "Currency": "₹",
                                "Location": r.get("Location", HOME_COUNTRY),
                                "Merchant": r.get("Merchant", MERCHANTS[0]),
                                "Device": r.get("Device", DEVICES[0]),
                                "Hour": int(r.get("Hour", 12)),
                                "DayOfWeek": int(r.get("DayOfWeek", datetime.now().weekday())),
                                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            score, status, reason = score_and_label(txn)
                            txn.update({"Score": score, "Status": status, "Reason": reason})
                            rows.append(txn)
                            st.session_state.txn_counter += 1
                        if rows:
                            st.session_state.transactions = pd.concat([st.session_state.transactions, pd.DataFrame(rows)], ignore_index=True)
                            st.success(f"Added {len(rows)} transactions to session.")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")

        st.markdown("---")
        st.subheader("Filters & Controls")
        currency_symbol = st.radio("Currency", ["₹", "$"], index=0, key="currency_selector")
        countries_filter = st.multiselect("Show countries", COUNTRIES, default=COUNTRIES[:4], key="filter_countries")
        merchants_filter = st.multiselect("Show merchants", MERCHANTS, default=MERCHANTS[:4], key="filter_merchants")
        amt_range = st.slider("Amount range (INR)", 0, 10000, (0,10000), step=100, key="amt_range")
        st.markdown("Adjust thresholds (demo scorer)")
        susp_th = st.slider("Suspicious threshold", 0.05, 0.9, 0.5, 0.05, key="susp_th")
        fraud_th = st.slider("Fraud threshold", 0.2, 1.0, 0.8, 0.05, key="fraud_th")

    with right:
        st.subheader("Live View & Insights")
        df_all = st.session_state.transactions.copy()
        if not df_all.empty:
            df_all["Amount"] = df_all["Amount"].astype(float)
            mask = (
                df_all["Location"].isin(countries_filter) &
                df_all["Merchant"].isin(merchants_filter) &
                df_all["Amount"].between(amt_range[0], amt_range[1])
            )
            df = df_all.loc[mask].copy()
        else:
            df = df_all

        col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
        total = len(df)
        frauds = df[df["Status"] == "Fraud"].shape[0] if not df.empty else 0
        suspicious = df[df["Status"] == "Suspicious"].shape[0] if not df.empty else 0
        money = df["Amount"].sum() if not df.empty else 0.0
        money_at_risk = df.loc[df["Status"]=="Fraud","Amount"].sum() if not df.empty else 0.0
        col_kpi1.metric("Total Txns", total)
        col_kpi2.metric("Fraud", frauds)
        col_kpi3.metric("Suspicious", suspicious)
        col_kpi4.metric("Money Processed", currency_fmt(money, currency_symbol))
        col_kpi5.metric("Money at Risk", currency_fmt(money_at_risk, currency_symbol))

        pie_pl = st.empty()
        bar_pl = st.empty()
        table_pl = st.empty()
        notif_pl = st.empty()

        if not df.empty:
            # Pie chart
            status_counts = df["Status"].value_counts().reindex(["Fraud","Suspicious","Safe"]).fillna(0)
            pie_df = pd.DataFrame({"Status": status_counts.index.astype(str),"Count": status_counts.values})
            if HAS_PLOTLY:
                fig_pie = px.pie(pie_df, names="Status", values="Count",
                                 color="Status",
                                 color_discrete_map={"Fraud":"red","Suspicious":"orange","Safe":"green"},
                                 title="Status distribution")
                pie_pl.plotly_chart(fig_pie, use_container_width=True, key="pie_main")
            else:
                pie_pl.bar_chart(pie_df.set_index("Status"))

            # Bar chart: high-risk countries
            fraud_by_country = (df[df["Status"]=="Fraud"].groupby("Location").size().reset_index(name="FraudCount"))
            if fraud_by_country.empty:
                fraud_by_country = pd.DataFrame({"Location":[],"FraudCount":[]})
            if HAS_PLOTLY:
                fig_bar = px.bar(fraud_by_country, x="Location", y="FraudCount", title="High-risk countries")
                bar_pl.plotly_chart(fig_bar, use_container_width=True, key="bar_main")
            else:
                bar_pl.bar_chart(fraud_by_country.set_index("Location"))

            # Transaction table
            df_display = df.copy()
            df_display["StatusLabel"] = df_display["Status"].apply(status_emoji)
            df_display = df_display[["TxnID","UserID","Amount","Currency","Location","Merchant","Device","Hour","StatusLabel","Score","Reason","Timestamp"]]
            table_pl.dataframe(df_display.style.apply(lambda x: color_rows(df_display), axis=None), use_container_width=True)

            # High-risk notifications
            highrisk = df[df["Status"]=="Fraud"].copy()
            if not highrisk.empty:
                notif_html = "<div class='blink'>⚠️ HIGH-RISK TRANSACTIONS DETECTED:</div><ul>"
                for _, row in highrisk.iterrows():
                    notif_html += f"<li><b>{row['TxnID']}</b> | {row['UserID']} | {row['Amount']} | {row['Reason']}</li>"
                notif_html += "</ul>"
                notif_pl.markdown(notif_html, unsafe_allow_html=True)
            else:
                notif_pl.info("No high-risk transactions currently.")

        else:
            pie_pl.info("No data yet. Add manual or upload CSV or run Real-Time Simulation.")
            bar_pl.empty()
            table_pl.empty()
            notif_pl.empty()

        # Export CSV
        if not st.session_state.transactions.empty:
            csv = st.session_state.transactions.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Export full session CSV", data=csv, file_name="session_transactions.csv", mime="text/csv", key="export_button")

# -------------------------
# REAL-TIME SIMULATION page
# -------------------------
elif page == "Real-Time Simulation":
    st.header("⚡ Real-Time Simulation")
    st.write("Stream live transactions with fraud & suspicious detection, plus a real-time dashboard summary.")

    left, right = st.columns([1,2])
    with left:
        start_btn = st.button("▶️ Start", key="start_stream")
        stop_btn = st.button("⏹ Stop", key="stop_stream")
        count = st.slider("How many transactions to stream (total)", 1, 500, 50, step=1, key="stream_count")
        speed = st.slider("Delay between transactions (seconds)", 0.05, 1.0, 0.4, step=0.05, key="stream_speed")
        currency_symbol = st.radio("Currency", ["₹","$"], index=0, key="stream_currency")
    with right:
        st.subheader("Live Stream (latest transactions below)")

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

    # Placeholders
    stream_kpi_pl = st.empty()
    stream_table_pl = st.empty()
    stream_pie_pl = st.empty()
    stream_bar_pl = st.empty()
    stream_alert_pl = st.empty()

    # --- Adjusted scoring thresholds for realistic fraud/susp ~10-15% ---
    susp_th = 0.85  # suspicious cutoff
    fraud_th = 0.93  # fraud cutoff

    if st.session_state.running:
        streamed = 0
        while streamed < count and st.session_state.running:
            # Generate txn
            txn = gen_txn(st.session_state.txn_counter, currency=currency_symbol)

            # Score txn with adjusted thresholds
            score, status, reason = score_and_label(txn, suspicious_th=susp_th, fraud_th=fraud_th)
            txn.update({"Score": score, "Status": status, "Reason": reason})
            st.session_state.txn_counter += 1

            # Add transaction
            st.session_state.transactions = pd.concat([st.session_state.transactions, pd.DataFrame([txn])], ignore_index=True)

            # Subset for display
            df = st.session_state.transactions.copy()
            df_display = df.tail(200)[["TxnID","UserID","Amount","Currency","Location","Merchant","Device","Hour","Status","Score","Reason","Timestamp"]]
            df_display["StatusLabel"] = df_display["Status"].apply(status_emoji)

            # === KPI Dashboard ===
            total_txns = len(df)
            susp_txns = (df["Status"]=="Suspicious").sum()
            fraud_txns = (df["Status"]=="Fraud").sum()
            susp_pct = (susp_txns/total_txns)*100 if total_txns>0 else 0
            fraud_pct = (fraud_txns/total_txns)*100 if total_txns>0 else 0
            money_at_risk = df.loc[df["Status"].isin(["Fraud","Suspicious"]), "Amount"].sum()

            kpi1, kpi2, kpi3 = stream_kpi_pl.columns(3)
            with kpi1:
                st.metric("📊 Total Transactions", total_txns)
            with kpi2:
                st.metric("🟠 Suspicious", f"{susp_txns} ({susp_pct:.1f}%)")
            with kpi3:
                st.metric("🔴 Fraud", f"{fraud_txns} ({fraud_pct:.1f}%)")

            kpi4, kpi5 = stream_kpi_pl.columns(2)
            with kpi4:
                st.metric("⚠️ Total Money at Risk", f"{currency_symbol}{money_at_risk:,.2f}")
            with kpi5:
                st.metric("✅ Safe Transactions", total_txns - susp_txns - fraud_txns)

            # === Live Table ===
            stream_table_pl.dataframe(df_display, use_container_width=True)

            # === Charts ===
            if HAS_PLOTLY:
                status_counts = df["Status"].value_counts().reindex(["Fraud","Suspicious","Safe"]).fillna(0)
                pie_df = pd.DataFrame({"Status": status_counts.index.astype(str),"Count": status_counts.values})
                fig_pie = px.pie(pie_df, names="Status", values="Count",
                                 color="Status", color_discrete_map={"Fraud":"red","Suspicious":"orange","Safe":"green"},
                                 title="Status Distribution (Live)")
                stream_pie_pl.plotly_chart(fig_pie, use_container_width=True, key=f"stream_pie_{streamed}")

                fraud_by_country = df[df["Status"]=="Fraud"].groupby("Location").size().reset_index(name="FraudCount")
                if fraud_by_country.empty:
                    fraud_by_country = pd.DataFrame({"Location":[],"FraudCount":[]})
                fig_bar = px.bar(fraud_by_country, x="Location", y="FraudCount", title="Fraud by Country (Live)")
                stream_bar_pl.plotly_chart(fig_bar, use_container_width=True, key=f"stream_bar_{streamed}")

            # === Alerts ===
            high_risk = df[df["Status"]=="Fraud"].copy()
            if not high_risk.empty:
                alert_text = f"⚠️ {len(high_risk)} High-Risk Transaction(s) detected!"
                stream_alert_pl.markdown(f"<div class='blink'>{alert_text}</div>", unsafe_allow_html=True)
            else:
                stream_alert_pl.info("No high-risk transactions currently.")

            streamed += 1
            time.sleep(speed)
    else:
        st.info("Stream is stopped. Click ▶️ Start to run the simulation.")

# -------------------------
# EXPLAINABILITY page
# -------------------------
elif page == "Explainability":
    st.header("🤖 Explainability & Model Notes")
    st.markdown("""
This demo uses a **transparent rule-based scorer** so the app runs without a heavy ML model.
You can swap `score_and_label()` with your trained `model.predict_proba()`.

**Scoring (demo):**
- +0.45 very high amount (≥ ₹5,000)
- +0.25 high amount (≥ ₹2,500)
- +0.30 foreign location
- +0.20 odd hours (before 06:00 or after 22:00)
- +0.10 device risk (POS/Desktop)
- +0.05 merchant risk (Dominos/Uber)
Score capped at 1.0

**Labels:**
- Fraud (score >= fraud threshold)
- Suspicious (score >= suspicious threshold)
- Safe otherwise

You can tune thresholds in Dashboard → Filters; production: replace with ML & SHAP for per-txn reasons.
""")

# -------------------------
# ABOUT page
# -------------------------
elif page == "About":
    st.header("About — NullOps")
    st.markdown(f"**Team**: {TEAM_NAME}  \n**Contact**: {CONTACT_EMAIL}")
    st.markdown("### Why this matters")
    st.write("""
    Frauders act fast — financial platforms need a low-latency, explainable system to reduce customer loss and operational cost.
    This prototype shows how to detect, explain, and export suspicious behavior in real-time.
    """)
    st.markdown("### What we built")
    st.write("""
    - Multi-input (manual, CSV, stream)  
    - Real-time stream simulation and live charts  
    - Explainable scoring (why flagged)  
    - Downloadable session CSV for reporting
    """)
