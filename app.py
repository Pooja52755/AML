import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import textwrap
import fraud_data
import graph_vis

# ─── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AML Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ───────────────────────────────────────────────────────────
st.html(textwrap.dedent("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp { background-color: #f1f5f9; }

    /* ── Header ── */
    .header-title { font-size: 24px; font-weight: 800; color: #0f172a; margin: 0; }
    .header-subtitle { font-size: 13px; color: #64748b; margin-top: 2px; }
    .header-timestamp { font-size: 13px; color: #64748b; }

    /* ── Badges ── */
    .badge-high {
        background:#fef2f2; color:#dc2626; border:1px solid #fecaca;
        padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700;
        display:inline-block;
    }
    .badge-medium {
        background:#fffbeb; color:#d97706; border:1px solid #fde68a;
        padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700;
        display:inline-block;
    }
    .badge-low {
        background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0;
        padding:3px 10px; border-radius:6px; font-size:11px; font-weight:700;
        display:inline-block;
    }

    /* ── Left Panel: Flagged Account Cards ── */
    .flagged-card {
        background:#ffffff; border:1.5px solid #e2e8f0; border-radius:10px;
        padding:12px 14px; margin-bottom:10px; cursor:pointer;
        transition: box-shadow 0.2s;
        border-left: 4px solid #e2e8f0;
    }
    .flagged-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .flagged-card-high { border-left-color: #ef4444 !important; }
    .flagged-card-medium { border-left-color: #f59e0b !important; }
    .flagged-card-selected { background:#eff6ff; border-color:#93c5fd; }
    .flagged-acc-id { font-weight:700; font-size:13px; color:#1e40af; }
    .flagged-pattern { font-size:11px; color:#64748b; margin-top:3px; }
    .risk-score-bar-bg {
        background:#f1f5f9; border-radius:4px; height:6px; margin-top:8px; overflow:hidden;
    }
    .risk-score-bar-fill {
        height:6px; border-radius:4px;
        background: linear-gradient(90deg, #f59e0b, #ef4444);
    }

    /* ── Center Panel ── */
    .center-card {
        background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
        padding:20px; box-shadow:0 1px 4px rgba(0,0,0,0.06);
    }
    .graph-network-btn {
        display:inline-flex; align-items:center; gap:8px;
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color:white; font-weight:700; font-size:13px;
        padding:10px 18px; border-radius:8px; cursor:pointer;
        border:none; margin-bottom:16px;
        box-shadow: 0 2px 8px rgba(37,99,235,0.35);
        transition: all 0.2s;
        text-decoration:none;
    }
    .graph-network-btn:hover {
        background: linear-gradient(135deg, #1e3a8a, #1d4ed8);
        box-shadow: 0 4px 14px rgba(37,99,235,0.45);
        transform: translateY(-1px);
    }

    /* ── XAI Explanation box ── */
    .xai-box {
        background:#fffbeb; border:1px solid #fde68a; border-radius:10px;
        padding:14px 16px; margin-top:16px;
    }
    .xai-box-high {
        background:#fff5f5; border:1px solid #fecaca;
    }
    .xai-title { font-weight:700; font-size:13px; color:#1e293b; margin-bottom:8px; }
    .xai-list { margin:0; padding-left:18px; font-size:12px; color:#334155; line-height:1.8; }

    /* ── AI Advisory banner ── */
    .ai-advisory {
        background:#f0f9ff; border:1px solid #bae6fd; border-radius:8px;
        padding:10px 14px; font-size:11.5px; color:#0369a1;
        margin-top:12px; display:flex; align-items:flex-start; gap:8px;
    }

    /* ── Human Decision Panel ── */
    .human-decision-panel {
        background: linear-gradient(135deg, #fffbeb 0%, #fff7ed 100%);
        border:1.5px solid #fbbf24; border-radius:12px;
        padding:18px 20px; margin-top:18px;
    }
    .human-decision-title {
        font-weight:800; font-size:14px; color:#92400e;
        display:flex; align-items:center; gap:8px; margin-bottom:6px;
    }
    .human-decision-disclaimer {
        font-size:11px; color:#78716c; background:#ffffff;
        border:1px solid #e7e5e4; border-radius:6px;
        padding:8px 12px; margin-top:10px;
    }

    /* ── Right Panel: Customer Profile ── */
    .profile-card {
        background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
        padding:18px; box-shadow:0 1px 4px rgba(0,0,0,0.06);
        height:100%;
    }
    .profile-empty {
        display:flex; flex-direction:column; align-items:center;
        justify-content:center; padding:40px 20px; text-align:center;
        color:#94a3b8;
    }
    .profile-metric-grid {
        display:grid; grid-template-columns:repeat(2,1fr); gap:10px; margin-bottom:14px;
    }
    .profile-metric-card {
        background:#f8fafc; border:1px solid #f1f5f9; border-radius:8px; padding:10px 12px;
    }
    .profile-metric-label { font-size:10px; color:#64748b; text-transform:uppercase; font-weight:600; }
    .profile-metric-val { font-size:15px; font-weight:700; color:#0f172a; margin-top:2px; }
    .kyc-verified { color:#16a34a; font-weight:700; }
    .kyc-flagged { color:#dc2626; font-weight:700; }
    .kyc-minimal { color:#d97706; font-weight:700; }
    .kyc-unverified { color:#dc2626; font-weight:700; }

    /* ── Behavior table ── */
    .custom-table { width:100%; border-collapse:collapse; font-size:12px; margin-top:8px; }
    .custom-table th {
        text-align:left; padding:8px 10px; background:#f8fafc;
        color:#64748b; font-weight:600; border-bottom:1px solid #e2e8f0;
    }
    .custom-table td { padding:8px 10px; border-bottom:1px solid #f1f5f9; color:#1e293b; }
    .change-high { color:#dc2626; font-weight:600; }
    .change-medium { color:#d97706; font-weight:600; }
    .change-low { color:#16a34a; font-weight:500; }

    /* ── Top banner ── */
    .top-banner {
        background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
        padding:16px 22px; display:flex; align-items:center; gap:20px;
        margin-bottom:20px; box-shadow:0 1px 3px rgba(0,0,0,0.05);
    }
    .top-banner-icon {
        width:46px; height:46px; background:#2563eb; border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        color:white; font-size:22px;
    }
    .banner-value { font-size:26px; font-weight:800; color:#0f172a; line-height:1.2; }
    .banner-subtext { font-size:12px; color:#16a34a; font-weight:600; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e2e8f0; }
    .sidebar-brand {
        display:flex; align-items:center; gap:10px;
        padding:10px 0 20px 0; border-bottom:1px solid #f1f5f9; margin-bottom:20px;
    }
    .sidebar-brand-title { font-size:16px; font-weight:800; color:#0f172a; }
    .sidebar-brand-subtitle { font-size:11px; color:#64748b; }

    /* ── Model info bar ── */
    .model-info-bar {
        background:#eff6ff; border:1px solid #dbeafe; border-radius:8px;
        padding:10px 14px; display:flex; justify-content:space-between;
        align-items:center; font-size:12px; color:#1e40af; margin-top:14px;
    }

    /* ── Section labels ── */
    .section-label {
        font-size:11px; font-weight:700; color:#64748b;
        text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;
    }
</style>
"""))

# ─── Session State Initialization ────────────────────────────────────────
if "selected_tx_id" not in st.session_state:
    st.session_state.selected_tx_id = "TX-10231"
if "selected_sub_tx" not in st.session_state:
    st.session_state.selected_sub_tx = None
if "human_decision_submitted" not in st.session_state:
    st.session_state.human_decision_submitted = {}
if "goto_graph" not in st.session_state:
    st.session_state.goto_graph = False

# ─── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.html(textwrap.dedent("""
    <div class="sidebar-brand">
        <div style="width:34px;height:34px;background:#2563eb;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;color:white;font-size:18px;">
            🛡️
        </div>
        <div>
            <div class="sidebar-brand-title">AML Fraud Detection</div>
            <div class="sidebar-brand-subtitle">Graph-based Transaction Monitoring</div>
        </div>
    </div>
    """))

    # If goto_graph flag is set, pre-select the graph page
    default_page_idx = 2 if st.session_state.goto_graph else 0
    page = st.radio(
        "Navigation",
        ["Dashboard", "Transactions", "Alerts / Graph Network", "Customers", "Reports", "Settings", "Help"],
        index=default_page_idx,
        label_visibility="collapsed"
    )
    if st.session_state.goto_graph and page == "Alerts / Graph Network":
        st.session_state.goto_graph = False

    st.html("<br><br><hr><div style='text-align:center;color:#94a3b8;font-size:11px;'>© 2025 AML System</div>")

# ─── Top Header ───────────────────────────────────────────────────────────
current_time = datetime.now().strftime("%I:%M:%S %p")
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.html(textwrap.dedent("""
    <div>
        <h1 class="header-title">AML Fraud Detection</h1>
        <div class="header-subtitle">Graph-based Transaction Monitoring</div>
    </div>
    """))
with col_h2:
    st.html(textwrap.dedent(f"""
    <div style="text-align:right;margin-top:10px;">
        <span class="header-timestamp">🕒 Last Updated: {current_time}</span>
    </div>
    """))

st.write("")

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":

    # Top Banner
    st.html(textwrap.dedent("""
    <div class="top-banner">
        <div class="top-banner-icon">📋</div>
        <div>
            <div style="font-size:12px;color:#64748b;font-weight:600;">Total Transactions Processed</div>
            <div class="banner-value">1,80,256</div>
            <div class="banner-subtext">↑ 12,430 today</div>
        </div>
        <div style="margin-left:40px;">
            <div style="font-size:12px;color:#64748b;font-weight:600;">High Risk Alerts</div>
            <div style="font-size:26px;font-weight:800;color:#dc2626;">3</div>
            <div style="font-size:12px;color:#dc2626;font-weight:600;">Require Human Review</div>
        </div>
        <div style="margin-left:40px;">
            <div style="font-size:12px;color:#64748b;font-weight:600;">Medium Risk</div>
            <div style="font-size:26px;font-weight:800;color:#d97706;">2</div>
            <div style="font-size:12px;color:#d97706;font-weight:600;">Under Monitoring</div>
        </div>
        <div style="margin-left:auto;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:10px 18px;text-align:center;">
            <div style="font-size:11px;color:#16a34a;font-weight:700;">SYSTEM STATUS</div>
            <div style="font-size:18px;font-weight:800;color:#16a34a;">● LIVE</div>
        </div>
    </div>
    """))

    # ── 3-Column Layout ─────────────────────────────────────────────────
    col_left, col_center, col_right = st.columns([1.0, 1.8, 1.2])

    # ════════════════════════════════════════════════════════════════════
    #  LEFT PANEL — Flagged Accounts
    # ════════════════════════════════════════════════════════════════════
    with col_left:
        st.html('<div class="section-label">🚨 Flagged Accounts</div>')

        all_txs = fraud_data.get_all_flagged_senders()
        # Sort: High first, then Medium, then Low
        order = {"High": 0, "Medium": 1, "Low": 2}
        all_txs.sort(key=lambda x: (order.get(x["risk"], 9), -x["risk_score"]))

        for tx in all_txs:
            acc = tx["account"]
            risk = tx["risk"]
            score = tx["risk_score"]
            pattern = tx["pattern"]
            is_sel = (tx["tx_id"] == st.session_state.selected_tx_id)

            risk_color = "#ef4444" if risk == "High" else "#f59e0b" if risk == "Medium" else "#22c55e"
            sel_bg = "#eff6ff" if is_sel else "#ffffff"
            sel_border = "#93c5fd" if is_sel else "#e2e8f0"
            bar_width = score

            profile = fraud_data.get_customer_profile(acc)
            name = profile.get("name", acc)

            st.html(textwrap.dedent(f"""
            <div class="flagged-card flagged-card-{risk.lower()}"
                 style="background:{sel_bg}; border-color:{sel_border};">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <div class="flagged-acc-id">{acc}</div>
                        <div style="font-size:11px;color:#475569;font-weight:500;margin-top:1px;">{name}</div>
                    </div>
                    <span class="badge-{risk.lower()}">{risk}</span>
                </div>
                <div class="flagged-pattern">📌 {pattern}</div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
                    <div class="risk-score-bar-bg" style="flex:1;margin-right:8px;">
                        <div class="risk-score-bar-fill" style="width:{bar_width}%;background:{risk_color};"></div>
                    </div>
                    <span style="font-size:11px;font-weight:700;color:{risk_color};">{score}/100</span>
                </div>
            </div>
            """))
            if st.button(f"Inspect →", key=f"left_btn_{tx['tx_id']}", use_container_width=True):
                st.session_state.selected_tx_id = tx["tx_id"]
                st.session_state.selected_sub_tx = None
                st.rerun()


    # ════════════════════════════════════════════════════════════════════
    #  CENTER PANEL — Fan-Out Transaction Table + XAI + Human Decision
    # ════════════════════════════════════════════════════════════════════
    with col_center:
        curr_tx = fraud_data.get_transaction_by_id(st.session_state.selected_tx_id)
        risk = curr_tx["risk"]
        risk_score = curr_tx["risk_score"]
        pattern = curr_tx["pattern"]

        # ── Header ──
        st.html(textwrap.dedent(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div>
                <div style="font-size:16px;font-weight:800;color:#0f172a;">
                    {curr_tx['tx_id']}
                    <span style="font-size:12px;font-weight:500;color:#64748b;margin-left:6px;">{curr_tx['pattern']}</span>
                </div>
                <div style="font-size:12px;color:#475569;margin-top:2px;">
                    Sender: <b>{curr_tx['account']}</b> ·
                    {curr_tx['timestamp']} · {curr_tx['payment_format']}
                </div>
            </div>
            <span class="badge-{risk.lower()}">{risk.upper()} RISK · {risk_score}/100</span>
        </div>
        """))

        # ── Graph Network Button ──
        if st.button("🕸️  Tap to View as Graph Network", key="graph_btn", use_container_width=False):
            st.session_state.goto_graph = True
            st.rerun()

        # ── Fan-Out Sub-Transaction Table ──
        st.html('<div class="section-label">Transaction Routing Flow</div>')

        fan_rows = fraud_data.get_fan_out_rows(st.session_state.selected_tx_id)
        df_fan = pd.DataFrame(fan_rows)
        df_fan.columns = ["Sub-TX ID", "To Account", "Amount (₹)", "Time", "Acc. Age (days)", "Status"]

        # Age coloring
        def age_color(age):
            if age <= 7:
                return "🔴"
            elif age <= 30:
                return "🟡"
            return "🟢"

        df_fan["Age Risk"] = df_fan["Acc. Age (days)"].apply(lambda a: age_color(a))
        df_display = df_fan[["Sub-TX ID", "To Account", "Amount (₹)", "Time", "Acc. Age (days)", "Age Risk", "Status"]]

        # Clickable table with row selection
        event = st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="fan_table"
        )

        # Handle row selection → populate right panel
        selected_rows = event.selection.get("rows", []) if event.selection else []
        if selected_rows:
            row_idx = selected_rows[0]
            sub_tx_data = fan_rows[row_idx]
            st.session_state.selected_sub_tx = sub_tx_data

        st.html("""
        <div style="font-size:10.5px;color:#94a3b8;margin-top:4px;">
            🖱️ Click a row to view the receiver's customer profile in the right panel.
            🔴 Age &lt;7d · 🟡 7–30d · 🟢 &gt;30d
        </div>
        """)

        # ── Why Flagged? XAI Box ──
        is_high = (risk == "High")
        xai_extra = "xai-box-high" if is_high else ""
        exps_html = "".join([f"<li>{e}</li>" for e in curr_tx["explanations"]])
        fraud_icon = "⚠️" if curr_tx["is_fraud"] else "ℹ️"
        fraud_label = "FRAUD DETECTED" if curr_tx["is_fraud"] else "SUSPICIOUS ACTIVITY"

        st.html(textwrap.dedent(f"""
        <div class="xai-box {xai_extra}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                <div class="xai-title">{fraud_icon} {fraud_label} — Why was this flagged?</div>
                <span class="badge-{risk.lower()}">{pattern}</span>
            </div>
            <ul class="xai-list">
                {exps_html}
            </ul>
            <div style="font-size:11px;color:#64748b;margin-top:10px;border-top:1px solid #fde68a;padding-top:8px;">
                Model: <b>{curr_tx['model_used']}</b> &nbsp;|&nbsp;
                Confidence: <b>{curr_tx['model_confidence']}</b>
            </div>
        </div>
        """))

        # ── AI Advisory ──
        st.html(textwrap.dedent("""
        <div class="ai-advisory">
            <span style="font-size:16px;">🤖</span>
            <span>
                <b>AI Advisory:</b> The analysis above is generated by an AI model to
                <b>support the fraud investigator's decision</b>. The AI does not block or approve
                transactions. All final decisions must be made by an authorized bank employee.
            </span>
        </div>
        """))

        # ── Human Decision Panel (HIGH RISK ONLY) ──
        if is_high:
            tx_key = curr_tx["tx_id"]
            already_submitted = st.session_state.human_decision_submitted.get(tx_key)

            st.html(textwrap.dedent(f"""
            <div class="human-decision-panel">
                <div class="human-decision-title">
                    🏦 Human Decision Required — {curr_tx['tx_id']}
                </div>
                <div style="font-size:12px;color:#92400e;margin-bottom:12px;">
                    Risk Score: <b>{risk_score}/100</b> — This transaction requires a bank investigator decision.
                </div>
            </div>
            """))

            if already_submitted:
                decision_val = already_submitted["decision"]
                color_map = {"✅ Approve Transaction": "#16a34a", "🚫 Block Transaction": "#dc2626", "📤 Escalate to Senior Investigator": "#d97706"}
                col = color_map.get(decision_val, "#16a34a")
                st.success(f"**Decision Recorded:** {decision_val}")
                st.info(f"**Investigator Notes:** {already_submitted['notes'] or '(none)'}")
                if st.button("Revise Decision", key=f"revise_{tx_key}"):
                    del st.session_state.human_decision_submitted[tx_key]
                    st.rerun()
            else:
                dec_col1, dec_col2 = st.columns([1, 1])
                with dec_col1:
                    decision = st.radio(
                        "**Investigator Decision**",
                        ["✅ Approve Transaction", "🚫 Block Transaction", "📤 Escalate to Senior Investigator"],
                        key=f"decision_{tx_key}",
                        index=2
                    )
                with dec_col2:
                    notes = st.text_area(
                        "**Investigator Notes**",
                        placeholder="Add reasoning, observations, or escalation notes...",
                        key=f"notes_{tx_key}",
                        height=112
                    )

                st.html("""
                <div class="human-decision-disclaimer">
                    ⚠️ <b>Disclaimer:</b> By submitting, you confirm this decision is made by an
                    authorized bank employee. The AI model provided supporting analysis only.
                    This action will be logged and audited.
                </div>
                """)

                if st.button(f"📋 Submit Decision for {tx_key}", key=f"submit_{tx_key}", type="primary", use_container_width=True):
                    st.session_state.human_decision_submitted[tx_key] = {
                        "decision": decision,
                        "notes": notes,
                        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
                    }
                    st.rerun()

    # ════════════════════════════════════════════════════════════════════
    #  RIGHT PANEL — Customer Profile (receiver, on row click)
    # ════════════════════════════════════════════════════════════════════
    with col_right:
        sub_tx = st.session_state.selected_sub_tx

        if sub_tx is None:
            st.html(textwrap.dedent("""
            <div class="profile-card">
                <div class="section-label">Customer Profile</div>
                <div class="profile-empty">
                    <div style="font-size:40px;margin-bottom:12px;">👤</div>
                    <div style="font-size:13px;font-weight:600;color:#64748b;">No row selected</div>
                    <div style="font-size:12px;margin-top:6px;color:#94a3b8;">
                        Click any transaction row in the center table to view the receiver's customer profile here.
                    </div>
                </div>
            </div>
            """))
        else:
            to_acc = sub_tx["to_account"]
            profile = fraud_data.get_receiver_profile(to_acc)
            risk_tier = profile.get("risk_tier", "Unknown")
            kyc = profile.get("kyc_status", "Unknown")

            tier_color = "#dc2626" if "High" in risk_tier else "#d97706" if "Medium" in risk_tier else "#16a34a"

            kyc_class = (
                "kyc-verified" if kyc == "Verified" else
                "kyc-flagged" if kyc == "Flagged" else
                "kyc-minimal" if "Minimal" in kyc else
                "kyc-unverified"
            )

            # Behavior comparison table rows (simplified for receivers)
            beh_rows_html = ""
            for r in [
                ("Total Incoming", profile.get("total_incoming", "—")),
                ("Total Outgoing", profile.get("total_outgoing", "—")),
                ("Unique Senders", str(profile.get("unique_senders", "—"))),
                ("Unique Receivers", str(profile.get("unique_receivers", "—"))),
                ("Avg Tx Amount", profile.get("avg_tx_amount", "—")),
                ("Total Transactions", profile.get("total_transactions", "—")),
            ]:
                beh_rows_html += f"<tr><td><b>{r[0]}</b></td><td style='text-align:right;'>{r[1]}</td></tr>"

            st.html(textwrap.dedent(f"""
            <div class="profile-card">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">
                    <div>
                        <div class="section-label">Receiver Profile</div>
                        <div style="font-size:16px;font-weight:800;color:#0f172a;">{profile.get('name', to_acc)}</div>
                        <div style="font-size:12px;color:#64748b;margin-top:2px;">{to_acc}</div>
                    </div>
                    <div style="text-align:right;">
                        <span style="background:{tier_color}20;color:{tier_color};border:1px solid {tier_color}44;
                                     padding:3px 10px;border-radius:6px;font-size:11px;font-weight:700;">
                            {risk_tier}
                        </span>
                    </div>
                </div>

                <div class="profile-metric-grid">
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">Account Type</div>
                        <div class="profile-metric-val">{profile.get('account_type', '—')}</div>
                    </div>
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">City</div>
                        <div class="profile-metric-val">{profile.get('city', '—')}</div>
                    </div>
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">KYC Status</div>
                        <div class="profile-metric-val {kyc_class}">{kyc}</div>
                    </div>
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">Open Since</div>
                        <div class="profile-metric-val" style="font-size:12px;">{profile.get('open_since', '—')}</div>
                    </div>
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">Devices</div>
                        <div class="profile-metric-val">{profile.get('device_count', '—')}</div>
                    </div>
                    <div class="profile-metric-card">
                        <div class="profile-metric-label">Account Age (days)</div>
                        <div class="profile-metric-val">{sub_tx.get('account_age_days', '—')}</div>
                    </div>
                </div>

                <div style="font-size:12px;font-weight:700;color:#1e293b;margin-bottom:6px;">Financial Summary</div>
                <table class="custom-table">
                    <tbody>
                        {beh_rows_html}
                    </tbody>
                </table>

                <div style="background:#fff5f5;border:1px solid #fecaca;border-radius:8px;
                            padding:10px 12px;margin-top:12px;font-size:12px;color:#7f1d1d;">
                    <b>🔍 AI Observation:</b> {profile.get('notes', 'No additional notes.')}
                </div>

                <div style="margin-top:12px;font-size:11px;color:#94a3b8;text-align:center;">
                    Sub-TX: <b>{sub_tx.get('sub_tx_id')}</b> ·
                    ₹{sub_tx.get('amount')} ·
                    {sub_tx.get('time')}
                </div>
            </div>
            """))



# ══════════════════════════════════════════════════════════════════════════════
#  TRANSACTIONS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Transactions":
    st.subheader("📊 All Transactions Log")
    df_all = fraud_data.get_transactions_df()
    c1, c2, c3 = st.columns(3)
    with c1:
        search_q = st.text_input("🔍 Search Transaction ID / Account", "")
    with c2:
        risk_filter = st.multiselect("Filter Risk Level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    with c3:
        min_amt = st.slider("Min Amount (₹)", 0, 1500000, 0)

    filtered_df = df_all[df_all["risk"].isin(risk_filter) & (df_all["amount"] >= min_amt)]
    if search_q:
        filtered_df = filtered_df[
            filtered_df["tx_id"].str.contains(search_q, case=False) |
            filtered_df["account"].str.contains(search_q, case=False)
        ]
    st.dataframe(
        filtered_df[["tx_id", "account", "timestamp", "amount_formatted", "risk", "risk_score", "pattern", "payment_format"]],
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
#  ALERTS / GRAPH NETWORK PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Alerts / Graph Network":
    st.subheader("🕸️ Graph Network Topology & Fraud Rings")
    st.markdown(
        "Visualizing transactional connections up to **2 hops**. "
        "Hover over any node to see the customer profile. "
        "Source (★) → Hop-1 Receivers (●) → Hop-2 Downstream Nodes (◆)"
    )

    # Pre-select the tx from dashboard if navigated via the graph button
    default_sel = st.session_state.selected_tx_id
    tx_options = [t["tx_id"] for t in fraud_data.TRANSACTIONS]
    default_idx = tx_options.index(default_sel) if default_sel in tx_options else 0

    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        sel_tx = st.selectbox(
            "Select Transaction Network to Inspect",
            tx_options,
            index=default_idx
        )
    with col_g2:
        show_2hop = st.checkbox("Show 2-Hop Neighbors", value=True)

    tx_info = fraud_data.get_transaction_by_id(sel_tx)

    # Info bar
    risk_col = "#dc2626" if tx_info["risk"] == "High" else "#d97706" if tx_info["risk"] == "Medium" else "#16a34a"
    st.html(textwrap.dedent(f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                padding:14px 18px;margin-bottom:16px;display:flex;gap:28px;align-items:center;">
        <div>
            <div style="font-size:11px;color:#64748b;font-weight:600;">PATTERN</div>
            <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info['pattern']}</div>
        </div>
        <div>
            <div style="font-size:11px;color:#64748b;font-weight:600;">RISK SCORE</div>
            <div style="font-size:14px;font-weight:700;color:{risk_col};">{tx_info['risk_score']}/100</div>
        </div>
        <div>
            <div style="font-size:11px;color:#64748b;font-weight:600;">MODEL</div>
            <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info['model_used']}</div>
        </div>
        <div>
            <div style="font-size:11px;color:#64748b;font-weight:600;">SENDER</div>
            <div style="font-size:14px;font-weight:700;color:#1e293b;">{tx_info['account']}</div>
        </div>
        <div>
            <div style="font-size:11px;color:#64748b;font-weight:600;">AMOUNT</div>
            <div style="font-size:14px;font-weight:700;color:#1e293b;">₹{tx_info['amount_formatted']}</div>
        </div>
    </div>
    """))

    fig_g = graph_vis.render_plotly_graph(sel_tx, include_2hop=show_2hop)
    st.plotly_chart(fig_g, use_container_width=True)

    # Legend explanation
    st.html(textwrap.dedent("""
    <div style="display:flex;gap:24px;justify-content:center;margin-top:4px;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <span style="width:14px;height:14px;background:#ef4444;border-radius:50%;display:inline-block;"></span>
            Source Account (Sender)
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <span style="width:14px;height:14px;background:#f59e0b;border-radius:50%;display:inline-block;"></span>
            Hop-1 Direct Receivers
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <span style="width:14px;height:14px;background:#94a3b8;border-radius:4px;display:inline-block;"></span>
            Hop-2 Downstream Nodes
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <span style="width:28px;height:2px;background:#ef4444;display:inline-block;"></span>
            Hop-1 Transfer (amount shown)
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;">
            <span style="width:28px;height:2px;background:#cbd5e1;border-bottom:2px dashed #cbd5e1;display:inline-block;"></span>
            Hop-2 Transfer
        </div>
    </div>
    """))

    st.markdown("---")
    # XAI explanations on graph page too
    st.html(textwrap.dedent(f"""
    <div style="background:#fff5f5;border:1px solid #fecaca;border-radius:10px;padding:14px 16px;">
        <div style="font-weight:700;font-size:13px;color:#dc2626;margin-bottom:8px;">⚠️ Why was this flagged?</div>
        <ul style="margin:0;padding-left:18px;font-size:12px;color:#334155;line-height:1.8;">
            {"".join([f"<li>{e}</li>" for e in tx_info["explanations"]])}
        </ul>
    </div>
    """))

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOMERS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Customers":
    st.subheader("👤 Customer Profiling & Network Risk")
    acc_sel = st.selectbox("Select Account ID", list(fraud_data.CUSTOMER_PROFILES.keys()))
    prof = fraud_data.get_customer_profile(acc_sel)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Name:** {prof.get('name', '—')}")
        st.markdown(f"**KYC Status:** {prof.get('kyc_status', '—')}")
        st.markdown(f"**City:** {prof.get('city', '—')}")
        st.markdown(f"**Account Type:** {prof.get('account_type', '—')}")
    with c2:
        st.markdown(f"**Risk Tier:** {prof.get('risk_tier', '—')}")
        st.markdown(f"**Open Since:** {prof.get('open_since', '—')}")
        st.markdown(f"**Last Login:** {prof.get('last_login', '—')}")
        st.markdown(f"**Devices:** {prof.get('device_count', '—')}")

    st.dataframe(pd.DataFrame(prof["behavior_summary"]), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  REMAINING PAGES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Reports":
    st.subheader("📄 Automated Compliance & Fraud Reports")
    st.markdown("Generate AI-driven regulatory reports for compliance and operations stakeholders.")
    if st.button("Generate Summary Compliance PDF Report"):
        st.success("✅ Report generated successfully for 1,80,256 transactions processed.")

elif page == "Settings":
    st.subheader("⚙️ System & Graph AI Settings")
    st.slider("GraphSAGE Risk Detection Sensitivity", 0.0, 1.0, 0.75)
    st.selectbox("Primary Baseline Model", ["GraphSAGE + XGBoost Ensemble", "Isolation Forest", "TabPFN", "GNN Convolutional"])

elif page == "Help":
    st.subheader("❓ Help & Documentation")
    st.markdown("""
### Fraud Patterns Explained:
- **Fan-Out**: Single source account transferring money to multiple newly opened receiver accounts in a short time frame.
- **Circular Money Flow**: Funds transferred in a ring (A ➔ B ➔ C ➔ A) to disguise origin.
- **Account Takeover**: Unexpected login locations and immediate high-value transfer out of a dormant account.
- **Structuring / Smurfing**: Breaking down large sums into multiple smaller transfers to stay below regulatory reporting limits.
- **Rapid Velocity**: Unusually high transaction frequency within a short time window.

### How to use the Dashboard:
1. Click a **Flagged Account** in the left panel to load its transactions.
2. Click any **row** in the transaction table to view the **receiver's customer profile** on the right.
3. Tap **"View as Graph Network"** to visualize the 2-hop transaction network.
4. For **High Risk** transactions, submit your decision in the **Human Decision Panel**.

### AI Advisory Disclaimer:
The AI model provides pattern analysis and risk scores to *support* the investigation. **Final decisions must always be made by an authorized bank employee.**
    """)
