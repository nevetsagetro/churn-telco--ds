import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from pathlib import Path

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight · Retention Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --border: #1f2d45;
    --accent: #00d4ff;
    --accent2: #ff6b6b;
    --accent3: #7c3aed;
    --text: #e2e8f0;
    --muted: #64748b;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

.stApp { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; letter-spacing: 0.1em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Syne', sans-serif; font-size: 2rem !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent3), var(--accent)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
    padding: 0.6rem 2rem !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    height: 100%;
    margin-bottom: 1rem;
}
.metric-card.danger { border-color: var(--danger); box-shadow: 0 0 20px rgba(239, 68, 68, 0.1); }
.metric-card.success { border-color: var(--success); box-shadow: 0 0 20px rgba(16, 185, 129, 0.1); }
.metric-card.warning { border-color: var(--warning); box-shadow: 0 0 20px rgba(245, 158, 11, 0.1); }
.metric-card.info { border-color: var(--accent); box-shadow: 0 0 20px rgba(0, 212, 255, 0.1); }

.churn-score {
    font-family: 'Syne', sans-serif;
    font-size: 4.5rem;
    font-weight: 800;
    line-height: 1;
    text-align: center;
}
.score-low { color: var(--success); }
.score-mid { color: var(--warning); }
.score-high { color: var(--danger); }

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-danger { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }
.badge-success { background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid var(--success); }
.badge-warning { background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid var(--warning); }

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
}

.app-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.02em;
}
.app-logo span { color: var(--accent); }

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--surface);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
}

hr { border-color: var(--border) !important; margin: 1.5rem 0; }

.info-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.875rem;
}
.info-row:last-child { border-bottom: none; }
.info-label { color: var(--muted); }
.info-value { color: var(--text); font-family: 'IBM Plex Mono', monospace; }

.recommendation-box {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.1), rgba(0, 212, 255, 0.05));
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models_dir = Path("models")
    churn_model    = joblib.load(models_dir / "phase2_churn_model.pkl")
    churn_scaler   = joblib.load(models_dir / "phase2_scaler.pkl")
    cluster_model  = joblib.load(models_dir / "phase3_kmeans.pkl")
    cluster_scaler = joblib.load(models_dir / "phase3_cluster_scaler.pkl")
    with open(models_dir / "phase2_metadata.json") as f:
        meta2 = json.load(f)
    with open(models_dir / "phase3_metadata.json") as f:
        meta3 = json.load(f)
    return churn_model, churn_scaler, cluster_model, cluster_scaler, meta2, meta3

try:
    churn_model, churn_scaler, cluster_model, cluster_scaler, meta2, meta3 = load_models()
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    MODEL_ERROR = str(e)

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
SERVICE_MAP = {
    'PhoneService'    : {'Yes': 1, 'No': 0},
    'MultipleLines'   : {'Yes': 1, 'No': 0, 'No phone service': 0},
    'InternetService' : {'DSL': 1, 'Fiber optic': 1, 'No': 0},
    'OnlineSecurity'  : {'Yes': 1, 'No': 0, 'No internet service': 0},
    'OnlineBackup'    : {'Yes': 1, 'No': 0, 'No internet service': 0},
    'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': 0},
    'TechSupport'     : {'Yes': 1, 'No': 0, 'No internet service': 0},
    'StreamingTV'     : {'Yes': 1, 'No': 0, 'No internet service': 0},
    'StreamingMovies' : {'Yes': 1, 'No': 0, 'No internet service': 0},
}

def build_churn_features(inputs, feature_cols, num_cols, scaler):
    df = pd.DataFrame([inputs])
    df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
    df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
    df['Num_Services']      = sum(df[c].map(m) for c, m in SERVICE_MAP.items() if c in df.columns)
    df['Has_Contract']      = (df['Contract'] != 'Month-to-month').astype(int)
    df['Tenure_x_Contract'] = df['tenure'] * df['Has_Contract']
    df['ChargeVelocity']    = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['BillRatio']         = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['ElecCheck']         = (df['PaymentMethod'] == 'Electronic check').astype(int)
    X_raw = pd.get_dummies(df.drop(columns=['TotalCharges'], errors='ignore'), drop_first=True)
    X_raw = X_raw.reindex(columns=feature_cols, fill_value=0)
    X_raw[num_cols] = scaler.transform(X_raw[num_cols])
    return X_raw

def build_cluster_features(inputs):
    df = pd.DataFrame([inputs])
    df['TotalCharges']   = df['MonthlyCharges'] * df['tenure']
    df['Num_Services']   = sum(df[c].map(m) for c, m in SERVICE_MAP.items() if c in df.columns)
    df['Has_Contract']   = (df['Contract'] != 'Month-to-month').astype(int)
    df['ChargeVelocity'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['ElecCheck']      = (df['PaymentMethod'] == 'Electronic check').astype(int)
    cluster_features = ['tenure','MonthlyCharges','TotalCharges','Num_Services',
                        'ChargeVelocity','Has_Contract','SeniorCitizen','ElecCheck']
    cluster_features = [c for c in cluster_features if c in df.columns]
    return df[cluster_features]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-logo">Churn<span>Sight</span></div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;font-size:0.72rem;margin-top:0.2rem;font-family:IBM Plex Mono,monospace;">RETENTION INTELLIGENCE · v1.0</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-label">👤 Customer Profile</div>', unsafe_allow_html=True)
    gender     = st.selectbox("Gender", ["Male", "Female"])
    senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner    = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure     = st.slider("Tenure (months)", 0, 72, 12)

    st.markdown("---")
    st.markdown('<div class="section-label">📱 Services</div>', unsafe_allow_html=True)
    phone       = st.selectbox("Phone Service", ["No", "Yes"])
    multi_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet    = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    if internet != "No":
        online_sec  = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
        online_bkp  = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
        device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_supp   = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
        stream_tv   = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
        stream_mv   = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])
    else:
        online_sec = online_bkp = device_prot = tech_supp = stream_tv = stream_mv = "No internet service"

    st.markdown("---")
    st.markdown('<div class="section-label">💳 Billing</div>', unsafe_allow_html=True)
    contract    = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless   = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment     = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_chg = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown('<h1 style="font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;margin-bottom:0.2rem;">Customer Churn Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#64748b;font-size:0.9rem;margin-bottom:1.5rem;">Predictive retention dashboard · Telco dataset · XGBoost + K-Means</p>', unsafe_allow_html=True)

if not MODELS_LOADED:
    st.error(f"⚠️ Models not found. Place your .pkl and .json files in a `models/` folder.\n\nError: {MODEL_ERROR}")
    st.info("Expected files:\n- `models/phase2_churn_model.pkl`\n- `models/phase2_scaler.pkl`\n- `models/phase2_metadata.json`\n- `models/phase3_kmeans.pkl`\n- `models/phase3_cluster_scaler.pkl`\n- `models/phase3_metadata.json`")
    st.stop()

# ── BUILD INPUTS ──────────────────────────────────────────────────────────────
inputs = {
    'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
    'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
    'PhoneService': phone, 'MultipleLines': multi_lines, 'InternetService': internet,
    'OnlineSecurity': online_sec, 'OnlineBackup': online_bkp,
    'DeviceProtection': device_prot, 'TechSupport': tech_supp,
    'StreamingTV': stream_tv, 'StreamingMovies': stream_mv,
    'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment,
    'MonthlyCharges': monthly_chg,
}

FEATURE_COLS  = meta2['feature_columns']
NUM_COLS      = [c for c in ['tenure', 'MonthlyCharges', 'Num_Services', 'Tenure_x_Contract'] if c in FEATURE_COLS]
THRESHOLD     = meta2.get('threshold', 0.17)
CAMPAIGN_COST = meta2.get('business_params', {}).get('campaign_cost', 50)
LTV_BENEFIT   = meta2.get('business_params', {}).get('ltv_benefit', 500)

# ── INFERENCE ─────────────────────────────────────────────────────────────────
X_churn    = build_churn_features(inputs, FEATURE_COLS, NUM_COLS, churn_scaler)
churn_prob = float(churn_model.predict_proba(X_churn)[0, 1])
churn_pred = int(churn_prob >= THRESHOLD)

X_clust_raw    = build_cluster_features(inputs)
X_clust_scaled = cluster_scaler.transform(X_clust_raw)
segment_id     = int(cluster_model.predict(X_clust_scaled)[0])

segment_names = meta3.get('segment_names', {})
# Handle both str and int keys
segment_name  = segment_names.get(str(segment_id), segment_names.get(segment_id, f"Segment {segment_id}"))

intervention_ev = churn_prob * LTV_BENEFIT - CAMPAIGN_COST
worth_targeting = intervention_ev > 0

# Risk helpers
risk_label = "LOW" if churn_prob < 0.25 else ("MEDIUM" if churn_prob < 0.5 else "HIGH")
risk_class  = "success" if churn_prob < 0.25 else ("warning" if churn_prob < 0.5 else "danger")
risk_color  = "#10b981" if churn_prob < 0.25 else ("#f59e0b" if churn_prob < 0.5 else "#ef4444")
score_class = "score-low" if churn_prob < 0.25 else ("score-mid" if churn_prob < 0.5 else "score-high")

num_services = sum(1 for c, m in SERVICE_MAP.items() if inputs.get(c) in ['Yes','DSL','Fiber optic'])
has_contract = contract != 'Month-to-month'
elec_check   = payment == 'Electronic check'

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  Prediction", "🧩  Segmentation", "💼  Business Case"])

# ══════════════════════════════════════════════════════════════
# TAB 1: PREDICTION
# ══════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown(f"""
        <div class="metric-card {risk_class}" style="text-align:center;">
            <div class="section-label" style="text-align:center;">CHURN PROBABILITY</div>
            <div class="churn-score {score_class}">{churn_prob*100:.1f}%</div>
            <div style="margin-top:0.6rem;">
                <span class="badge badge-{risk_class}">{risk_label} RISK</span>
            </div>
            <div style="margin-top:0.8rem;color:#64748b;font-size:0.75rem;font-family:IBM Plex Mono,monospace;">
                THRESHOLD {THRESHOLD:.4f} · {"⚠ CHURN" if churn_pred else "✓ RETAIN"}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card info">
            <div class="section-label">KEY RISK SIGNALS</div>
            <div class="info-row">
                <span class="info-label">Contract</span>
                <span class="info-value" style="color:{'#10b981' if has_contract else '#ef4444'}">
                    {'✓ Long-term' if has_contract else '⚠ Monthly'}
                </span>
            </div>
            <div class="info-row">
                <span class="info-label">Services</span>
                <span class="info-value">{num_services} / 9</span>
            </div>
            <div class="info-row">
                <span class="info-label">Payment</span>
                <span class="info-value" style="color:{'#ef4444' if elec_check else '#10b981'}">
                    {'⚠ Elec. Check' if elec_check else '✓ Stable'}
                </span>
            </div>
            <div class="info-row">
                <span class="info-label">Tenure</span>
                <span class="info-value">{tenure} months</span>
            </div>
            <div class="info-row">
                <span class="info-label">Monthly Charges</span>
                <span class="info-value">${monthly_chg:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            number={'suffix': '%', 'font': {'size': 42, 'family': 'Syne', 'color': risk_color}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#64748b', 'tickfont': {'color': '#64748b', 'size': 10}},
                'bar': {'color': risk_color, 'thickness': 0.35},
                'bgcolor': '#1a2235', 'borderwidth': 0,
                'steps': [
                    {'range': [0, 25],  'color': 'rgba(16,185,129,0.1)'},
                    {'range': [25, 50], 'color': 'rgba(245,158,11,0.1)'},
                    {'range': [50, 100],'color': 'rgba(239,68,68,0.1)'},
                ],
                'threshold': {'line': {'color': '#fff', 'width': 2}, 'thickness': 0.75, 'value': THRESHOLD * 100}
            },
            title={'text': 'Risk Gauge · white line = decision threshold', 'font': {'size': 11, 'color': '#64748b', 'family': 'IBM Plex Mono'}}
        ))
        gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e2e8f0'}, height=260, margin=dict(t=40, b=5, l=30, r=30)
        )
        st.plotly_chart(gauge, use_container_width=True)

        # Risk factor bars — aligned with real XGBoost top features
        # Top features: Has_Contract, BillRatio, InternetService_Fiber optic, tenure, MonthlyCharges
        bill_ratio = monthly_chg / (monthly_chg * tenure + 1)  # proxy: high when tenure is low
        drivers = {
            'No Long-Term Contract':     0.38 if not has_contract else 0.04,
            'High Bill Ratio (new cust)':min(0.32, bill_ratio * 4),
            'Fiber Optic Internet':      0.24 if internet == 'Fiber optic' else 0.03,
            'Low Tenure':                max(0.02, (1 - min(tenure, 72) / 72) * 0.26),
            'High Monthly Charges':      min(0.28, (monthly_chg - 18) / (120 - 18) * 0.28),
            'Electronic Check Payment':  0.18 if elec_check else 0.03,
            'No Online Security':        0.14 if online_sec == 'No' else 0.02,
            'No Tech Support':           0.11 if tech_supp == 'No' else 0.02,
        }
        sorted_d = dict(sorted(drivers.items(), key=lambda x: x[1], reverse=True))
        bar_colors = ['#ef4444' if v > 0.15 else '#f59e0b' if v > 0.08 else '#10b981' for v in sorted_d.values()]

        fig_b = go.Figure(go.Bar(
            x=list(sorted_d.values()), y=list(sorted_d.keys()),
            orientation='h', marker_color=bar_colors, marker_line_width=0,
        ))
        fig_b.update_layout(
            title={'text': 'Risk Factor Decomposition', 'font': {'size': 12, 'color': '#64748b', 'family': 'IBM Plex Mono'}},
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e2e8f0', 'family': 'Inter'},
            xaxis={'showgrid': False, 'zeroline': False, 'tickformat': '.0%', 'color': '#64748b'},
            yaxis={'showgrid': False, 'color': '#e2e8f0', 'tickfont': {'size': 11}},
            height=270, margin=dict(t=35, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_b, use_container_width=True)

    # Recommendations
    if churn_pred == 1:
        recs = []
        if not has_contract: recs.append("🔒 <b>Offer a contract upgrade</b> — strongest single churn reducer (1 or 2-year deal with loyalty discount)")
        if elec_check:        recs.append("💳 <b>Migrate to auto-payment</b> — electronic check is a churn proxy; incentivize bank transfer/credit card")
        if num_services < 3:  recs.append("📦 <b>Bundle upsell campaign</b> — customers with 4+ services have ~40% lower churn")
        if tenure < 12:       recs.append("🎁 <b>First-year loyalty program</b> — months 0–12 are the highest-risk window")
        rec_html = "<br>".join(recs) if recs else "⚡ Proactive outreach with personalized retention offer recommended."
        st.markdown(f"""
        <div class="recommendation-box">
            <div class="section-label">💡 RECOMMENDED INTERVENTIONS</div>
            <div style="margin-top:0.5rem;line-height:2;font-size:0.9rem;">{rec_html}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="recommendation-box" style="border-color:rgba(16,185,129,0.3);background:linear-gradient(135deg,rgba(16,185,129,0.05),rgba(0,212,255,0.03));">
            <div class="section-label">✅ RETENTION STATUS</div>
            <div style="margin-top:0.5rem;font-size:0.9rem;">
                Customer is <b>stable</b>. No immediate intervention required.
                Consider a satisfaction touchpoint at month <b>{tenure + 6}</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2: SEGMENTATION
# ══════════════════════════════════════════════════════════════
with tab2:
    seg_color_map = {
        'High-Value At-Risk': '#ef4444',
        'Low-Value At-Risk':  '#f59e0b',
        'Loyal Anchors':      '#10b981',
        'New & Vulnerable':   '#f97316',
        'Premium Stable':     '#00d4ff',
        'Core Stable':        '#7c3aed',
    }
    seg_color = seg_color_map.get(segment_name, '#64748b')

    seg_defs = {
        'High-Value At-Risk': ('🚨', 'High monthly spend combined with elevated churn probability. Prime candidate for immediate retention with a premium offer.'),
        'Low-Value At-Risk':  ('⚠️', 'Lower spend but high churn risk. Evaluate cost of retention vs. LTV carefully before investing.'),
        'Loyal Anchors':      ('⚓', 'Long-tenured, contracted customers. Focus on upsell and deepening product adoption rather than retention.'),
        'New & Vulnerable':   ('🌱', 'Early lifecycle, not yet committed. Critical 90-day onboarding window — high-touch nurture required.'),
        'Premium Stable':     ('👑', 'High ARPU with stable behavior. Protect with exclusive loyalty perks and proactive health checks.'),
        'Core Stable':        ('🔵', 'Reliable base revenue. Maintain satisfaction with periodic check-ins and feature adoption nudges.'),
    }
    emoji, desc = seg_defs.get(segment_name, ('🔵', 'Standard customer segment.'))

    col_s1, col_s2 = st.columns([1, 1], gap="large")

    with col_s1:
        st.markdown(f"""
        <div class="metric-card" style="border-color:{seg_color};box-shadow:0 0 24px {seg_color}22;text-align:center;">
            <div class="section-label" style="text-align:center;">CUSTOMER SEGMENT</div>
            <div style="font-size:2.8rem;margin:0.3rem 0;">{emoji}</div>
            <div style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;color:{seg_color};">
                {segment_name}
            </div>
            <div style="color:#64748b;font-size:0.75rem;font-family:IBM Plex Mono,monospace;margin-top:0.4rem;">
                CLUSTER ID: {segment_id}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card info">
            <div class="section-label">SEGMENT DEFINITION</div>
            <div style="margin-top:0.5rem;font-size:0.9rem;line-height:1.7;color:#e2e8f0;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        charge_velocity = monthly_chg / (tenure + 1)
        categories = ['Tenure', 'Monthly\nCharges', 'Services\nAdopted', 'Contract\nCommit', 'Spend\nVelocity']
        vals_customer = [
            tenure / 72,
            (monthly_chg - 18) / (120 - 18),
            num_services / 9,
            1.0 if has_contract else 0.0,
            min(1.0, charge_velocity / 30),
        ]
        seg_avg_map = {
            'High-Value At-Risk': [0.35, 0.80, 0.55, 0.10, 0.70],
            'Low-Value At-Risk':  [0.25, 0.35, 0.35, 0.10, 0.55],
            'Loyal Anchors':      [0.80, 0.62, 0.72, 0.90, 0.20],
            'New & Vulnerable':   [0.12, 0.48, 0.28, 0.05, 0.90],
            'Premium Stable':     [0.60, 0.90, 0.78, 0.85, 0.38],
            'Core Stable':        [0.50, 0.52, 0.58, 0.58, 0.33],
        }
        seg_avg = seg_avg_map.get(segment_name, [0.5]*5)

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=seg_avg + [seg_avg[0]], theta=categories + [categories[0]],
            fill='toself', fillcolor=f'{seg_color}22',
            line=dict(color=seg_color, width=2, dash='dash'),
            name=f'{segment_name} (avg)',
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_customer + [vals_customer[0]], theta=categories + [categories[0]],
            fill='toself', fillcolor='rgba(0,212,255,0.15)',
            line=dict(color='#00d4ff', width=2),
            name='This customer',
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0,1], tickfont={'color':'#64748b','size':8}, gridcolor='#1f2d45', linecolor='#1f2d45'),
                angularaxis=dict(tickfont={'color':'#e2e8f0','size':10}, linecolor='#1f2d45', gridcolor='#1f2d45'),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color':'#e2e8f0','family':'Inter'},
            title={'text':'Customer vs Segment Profile','font':{'size':12,'color':'#64748b','family':'IBM Plex Mono'}},
            legend=dict(font={'size':11}, bgcolor='rgba(0,0,0,0)'),
            height=380, margin=dict(t=50,b=20,l=60,r=60)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # All segments table
    st.markdown("---")
    st.markdown('<div class="section-label">ALL SEGMENTS OVERVIEW</div>', unsafe_allow_html=True)
    seg_churn_rates = meta3.get('segment_churn_rates', {})
    seg_sizes       = meta3.get('segment_sizes', {})

    rows = []
    for sid, sname in segment_names.items():
        cr = float(seg_churn_rates.get(sid, seg_churn_rates.get(int(sid) if isinstance(sid,str) else str(sid), 0)))
        sz = int(seg_sizes.get(sid, seg_sizes.get(int(sid) if isinstance(sid,str) else str(sid), 0)))
        rows.append({
            'Active': '▶' if str(sid) == str(segment_id) else '',
            'Segment Name': sname,
            'Cluster': sid,
            'Churn Rate': f"{cr*100:.1f}%",
            'Size (customers)': sz,
            'Risk Level': '🔴 High' if cr > 0.40 else ('🟡 Medium' if cr > 0.20 else '🟢 Low'),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 3: BUSINESS CASE
# ══════════════════════════════════════════════════════════════
with tab3:
    ltv_est = monthly_chg * 24

    col_b1, col_b2, col_b3 = st.columns(3, gap="medium")
    with col_b1:
        ev_class = "success" if worth_targeting else "danger"
        ev_sign  = "+" if intervention_ev > 0 else ""
        ev_col   = "#10b981" if worth_targeting else "#ef4444"
        st.markdown(f"""
        <div class="metric-card {ev_class}">
            <div class="section-label">NET EXPECTED VALUE</div>
            <div style="font-family:Syne,sans-serif;font-size:2.6rem;font-weight:800;color:{ev_col};">
                {ev_sign}${intervention_ev:.0f}
            </div>
            <div style="color:#64748b;font-size:0.8rem;margin-top:0.3rem;">per retention campaign</div>
            <div style="margin-top:1rem;color:#e2e8f0;font-size:0.9rem;">
                <b>{'✅ Worth targeting' if worth_targeting else '❌ Skip — negative ROI'}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b2:
        st.markdown(f"""
        <div class="metric-card info">
            <div class="section-label">ESTIMATED 2-YR LTV</div>
            <div style="font-family:Syne,sans-serif;font-size:2.6rem;font-weight:800;color:#00d4ff;">
                ${ltv_est:,.0f}
            </div>
            <div style="color:#64748b;font-size:0.8rem;margin-top:0.3rem;">24-month revenue proxy</div>
            <div style="margin-top:1rem;">
                <div class="info-row"><span class="info-label">Monthly</span><span class="info-value">${monthly_chg:.2f}</span></div>
                <div class="info-row"><span class="info-label">Annual</span><span class="info-value">${monthly_chg*12:,.0f}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b3:
        rev_at_risk = ltv_est * churn_prob
        st.markdown(f"""
        <div class="metric-card warning">
            <div class="section-label">REVENUE AT RISK</div>
            <div style="font-family:Syne,sans-serif;font-size:2.6rem;font-weight:800;color:#f59e0b;">
                ${rev_at_risk:,.0f}
            </div>
            <div style="color:#64748b;font-size:0.8rem;margin-top:0.3rem;">probability-weighted exposure</div>
            <div style="margin-top:1rem;">
                <div class="info-row"><span class="info-label">Churn Prob</span><span class="info-value">{churn_prob*100:.1f}%</span></div>
                <div class="info-row"><span class="info-label">Campaign Cost</span><span class="info-value">${CAMPAIGN_COST}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col_w1, col_w2 = st.columns(2, gap="large")

    with col_w1:
        st.markdown('<div class="section-label">ROI WATERFALL</div>', unsafe_allow_html=True)
        fig_wf = go.Figure(go.Waterfall(
            orientation='v',
            measure=['absolute', 'relative', 'total'],
            x=['Expected Benefit\n(prob × LTV)', 'Campaign Cost', 'Net EV'],
            y=[churn_prob * LTV_BENEFIT, -CAMPAIGN_COST, 0],
            connector={'line': {'color': '#1f2d45', 'width': 1}},
            increasing={'marker': {'color': '#10b981'}},
            decreasing={'marker': {'color': '#ef4444'}},
            totals={'marker': {'color': '#00d4ff'}},
            text=[f'${churn_prob*LTV_BENEFIT:.0f}', f'-${CAMPAIGN_COST}', f'${intervention_ev:.0f}'],
            textposition='outside',
            textfont={'color': '#e2e8f0', 'size': 13, 'family': 'IBM Plex Mono'},
        ))
        fig_wf.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color':'#e2e8f0','family':'Inter'},
            yaxis={'showgrid':False,'zeroline':True,'zerolinecolor':'#1f2d45','color':'#64748b','tickprefix':'$'},
            xaxis={'showgrid':False,'color':'#64748b'},
            height=340, margin=dict(t=20,b=30,l=20,r=20)
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    with col_w2:
        st.markdown('<div class="section-label">EXPECTED VALUE CURVE</div>', unsafe_allow_html=True)
        p_range  = np.linspace(0, 1, 200)
        ev_range = p_range * LTV_BENEFIT - CAMPAIGN_COST
        breakeven_p = CAMPAIGN_COST / LTV_BENEFIT

        fig_ev = go.Figure()
        mask_pos = ev_range >= 0
        mask_neg = ev_range < 0
        fig_ev.add_trace(go.Scatter(
            x=p_range[mask_pos], y=ev_range[mask_pos],
            fill='tozeroy', fillcolor='rgba(16,185,129,0.12)',
            line=dict(color='#10b981', width=2), name='Positive EV', showlegend=False,
        ))
        fig_ev.add_trace(go.Scatter(
            x=p_range[mask_neg], y=ev_range[mask_neg],
            fill='tozeroy', fillcolor='rgba(239,68,68,0.08)',
            line=dict(color='#ef4444', width=2), name='Negative EV', showlegend=False,
        ))
        fig_ev.add_vline(x=churn_prob, line_color='#00d4ff', line_width=2,
                         annotation_text=f' This customer ({churn_prob*100:.1f}%)',
                         annotation_font_color='#00d4ff', annotation_font_size=11)
        fig_ev.add_vline(x=breakeven_p, line_color='#f59e0b', line_width=1, line_dash='dash',
                         annotation_text=f' Break-even ({breakeven_p*100:.1f}%)',
                         annotation_font_color='#f59e0b', annotation_font_size=10)
        fig_ev.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color':'#e2e8f0','family':'Inter'},
            xaxis={'title':'Churn Probability','color':'#64748b','tickformat':'.0%','showgrid':False},
            yaxis={'title':'Expected Value ($)','color':'#64748b','showgrid':False,'zeroline':True,'zerolinecolor':'#1f2d45'},
            height=340, margin=dict(t=20,b=40,l=20,r=20)
        )
        st.plotly_chart(fig_ev, use_container_width=True)

    # Model card
    st.markdown("---")
    st.markdown('<div class="section-label">MODEL PERFORMANCE CARD</div>', unsafe_allow_html=True)
    m_cols = st.columns(4, gap="medium")
    kpis = [
        ('Best Model',   meta2.get('best_model', 'N/A')),
        ('ROC-AUC',      f"{meta2.get('test_auc', meta2.get('best_auc', 0)):.4f}"),
        ('Decision Threshold', f"{THRESHOLD:.4f}"),
        ('Training Samples',   f"{meta2.get('dataset_size', 'N/A'):,}" if isinstance(meta2.get('dataset_size'), int) else str(meta2.get('dataset_size','N/A'))),
    ]
    for col, (label, value) in zip(m_cols, kpis):
        with col:
            st.metric(label, value)