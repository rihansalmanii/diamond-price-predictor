from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

# --- Configuration & Paths ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diamond_rf_model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"

st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern CSS Styling ---
st.markdown(
    """
    <style>
        /* Global Background */
        .main {
            background-color: #0f172a; /* Dark Slate 900 */
            color: #f8fafc;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Header Styling */
        .header-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.5rem;
        }
        .header-subtitle {
            font-size: 1.1rem;
            color: #94a3b8;
            margin-bottom: 2rem;
        }

        /* Section Labels (PHYSICAL PROPERTIES, etc.) */
        .section-label {
            font-size: 0.75rem;
            font-weight: 700;
            color: #64748b; /* Muted Blue-Gray */
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 1rem;
            display: block;
            margin-top: 0.5rem;
        }

        /* Result Card Styling */
        .result-card {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.3);
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .price-label {
            font-size: 0.85rem;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
        }
        .price-value {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.5rem;
        }
        .price-range {
            font-size: 0.9rem;
            opacity: 0.8;
            background: rgba(0,0,0,0.2);
            display: inline-block;
            padding: 0.25rem 0.8rem;
            border-radius: 20px;
            margin-top: 0.5rem;
        }

        /* Sidebar Styling */
        .metric-box {
            background: #1e293b;
            padding: 0.75rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            text-align: center;
            border: 1px solid #334155;
        }
        .metric-val {
            font-size: 1.25rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .metric-lbl {
            font-size: 0.7rem;
            color: #94a3b8;
            text-transform: uppercase;
        }
        
        /* --- INPUT FIELD STYLING (The Look You Want) --- */
        
        /* Force labels to be White and sit above inputs */
        label {
            color: #ffffff !important;
            font-size: 0.9rem !important;
            font-weight: 500 !important;
            margin-bottom: 0.4rem !important;
        }

        /* The Input Container (Dark Box) */
        div[data-baseweb="base-input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background-color: #171C26 !important; /* Dark Slate 800 */
            border-radius: 8px !important;
            color: #ffffff !important;
            height: 48px !important; /* Taller, modern feel */
        }

        /* The Actual Text Inside */
        input {
            color: #ffffff !important;
            font-weight: 500 !important;
        }

        /* Hover Effects */
        div[data-baseweb="base-input"]:hover > div,
        div[data-baseweb="select"]:hover > div {
            background-color: #253346 !important;
            border-color: #475569 !important;
        }

        /* Dropdown Menu Styling */
        div[data-baseweb="menu"] {
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
        }
        div[data-baseweb="option"] {
            color: #f8fafc !important;
        }
        div[data-baseweb="option"]:hover {
            background-color: #334155 !important;
        }

        /* Help Icon Color */
        button[kind="help"] svg {
            fill: #94a3b8 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load Assets ---
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata

try:
    model, metadata = load_assets()
except FileNotFoundError:
    st.error("⚠️ Model files missing.")
    st.stop()

# --- Prepare Mappings ---
CUT_MAP = {name: i for i, name in enumerate(metadata["cut_order"])}
COLOR_MAP = {name: i for i, name in enumerate(metadata["color_order"])}
CLARITY_MAP = {name: i for i, name in enumerate(metadata["clarity_order"])}

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ℹ️ Model Info")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{metadata['r2_score']:.3f}</div>
            <div class="metric-lbl">R² Score</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{metadata['training_rows']:,}</div>
            <div class="metric-lbl">Rows</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 Grade Guide")
    st.markdown("""
    **Cut:** Ideal > Premium > Very Good > Good > Fair  
    **Color:** D (Colorless) → J (Light Yellow)  
    **Clarity:** IF (Flawless) → VVS → VS → SI → I1
    """, help="Standard GIA Grading Scale")
    
    st.caption("Model: Random Forest Regressor\nTarget: USD")

# --- Main Content ---
st.markdown('<div class="header-title">💎 Diamond Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Enter specifications to estimate market value.</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    with st.form("prediction_form", clear_on_submit=False):
        
        # Section 1
        st.markdown('<span class="section-label">Physical Properties</span>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            carat = st.number_input("Carat Weight", min_value=0.10, max_value=6.00, value=1.00, step=0.01)
        with c2:
            length_mm = st.number_input("Length (mm)", min_value=1.0, max_value=15.0, value=6.40, step=0.01)
        with c3:
            width_mm = st.number_input("Width (mm)", min_value=1.0, max_value=14.99, value=6.40, step=0.01)
        
        # Section 2
        st.markdown('<span class="section-label">Quality Grades</span>', unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3)
        with q1:
            cut = st.selectbox("Cut Grade", metadata["cut_order"], index=4)
        with q2:
            color = st.selectbox("Color Grade", list(reversed(metadata["color_order"])), index=3) 
        with q3:
            clarity = st.selectbox("Clarity Grade", metadata["clarity_order"], index=3)

        # Section 3
        st.markdown('<span class="section-label">Proportions</span>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        with p1:
            depth_percent = st.number_input("Depth (%)", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
        with p2:
            table = st.number_input("Table (%)", min_value=40.0, max_value=100.0, value=57.0, step=0.1)
        with p3:
            depth_mm = st.number_input("Physical Depth (mm)", min_value=1.0, max_value=14.99, value=4.00, step=0.01)

        submitted = st.form_submit_button("Estimate Price", type="primary", use_container_width=True)

with col_right:
    if not submitted:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 1rem; color: #475569;">
            <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;"></div>
            <p>Adjust parameters on the left<br>to see the prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        input_data = pd.DataFrame(
            [[carat, CUT_MAP[cut], COLOR_MAP[color], CLARITY_MAP[clarity], depth_percent, table, length_mm, width_mm, depth_mm]],
            columns=metadata["feature_names"],
        )

        prediction = max(float(model.predict(input_data)[0]), 0.0)
        lower = max(prediction * 0.90, 0.0)
        upper = prediction * 1.10

        st.markdown(
            f"""
            <div class="result-card">
              <div class="price-label">Estimated Market Value</div>
              <div class="price-value">${prediction:,.0f}</div>
              <div class="price-range">Range: ${lower:,.0f} – ${upper:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.info("💡 Indicative range based on model confidence. Actual retail prices may vary.", icon="ℹ️")

st.divider()
st.caption("Built with Streamlit • Model: Random Forest Regressor")