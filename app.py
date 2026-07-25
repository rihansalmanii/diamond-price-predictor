from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diamond_rf_model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"

st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 2rem;}
    .hero {
        padding: 2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(124,58,237,.16), rgba(14,165,233,.12));
        border: 1px solid rgba(148,163,184,.25);
        margin-bottom: 1.4rem;
    }
    .hero h1 {margin: 0; font-size: 2.45rem;}
    .hero p {margin: .65rem 0 0; font-size: 1.05rem; opacity: .82;}
    .result-card {
        padding: 1.5rem;
        border-radius: 18px;
        border: 1px solid rgba(148,163,184,.25);
        background: rgba(124,58,237,.07);
        text-align: center;
    }
    .result-label {font-size: .95rem; opacity: .72;}
    .result-price {font-size: 2.35rem; font-weight: 750; margin: .2rem 0;}
    .small-note {font-size: .88rem; opacity: .68;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata

try:
    model, metadata = load_assets()
except FileNotFoundError:
    st.error("Model files are missing. Keep diamond_rf_model.joblib and model_metadata.json beside app.py.")
    st.stop()

CUT_MAP = {name: i for i, name in enumerate(metadata["cut_order"])}
COLOR_MAP = {name: i for i, name in enumerate(metadata["color_order"])}
CLARITY_MAP = {name: i for i, name in enumerate(metadata["clarity_order"])}

st.markdown(
    """
    <div class="hero">
      <h1>💎 Diamond Price Predictor</h1>
      <p>Estimate a diamond's price from its carat, quality grades, proportions, and physical dimensions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.45, 0.75], gap="large")

with left:
    st.subheader("Diamond details")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            carat = st.number_input("Carat", min_value=0.10, max_value=6.00, value=1.00, step=0.01)
            cut = st.selectbox("Cut", metadata["cut_order"], index=4)
            color = st.selectbox("Color", list(reversed(metadata["color_order"])), index=3)
        with c2:
            clarity = st.selectbox("Clarity", metadata["clarity_order"], index=3)
            depth_percent = st.number_input("Depth (%)", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
            table = st.number_input("Table (%)", min_value=40.0, max_value=100.0, value=57.0, step=0.1)
        with c3:
            length_mm = st.number_input("Length (mm)", min_value=1.0, max_value=15.0, value=6.40, step=0.01)
            width_mm = st.number_input("Width (mm)", min_value=1.0, max_value=14.99, value=6.40, step=0.01)
            depth_mm = st.number_input("Physical depth (mm)", min_value=1.0, max_value=14.99, value=4.00, step=0.01)

        submitted = st.form_submit_button("Predict price", type="primary", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame(
            [[
                carat,
                CUT_MAP[cut],
                COLOR_MAP[color],
                CLARITY_MAP[clarity],
                depth_percent,
                table,
                length_mm,
                width_mm,
                depth_mm,
            ]],
            columns=metadata["feature_names"],
        )

        prediction = max(float(model.predict(input_data)[0]), 0.0)
        lower = max(prediction * 0.90, 0.0)
        upper = prediction * 1.10

        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-label">Estimated price</div>
              <div class="result-price">${prediction:,.0f}</div>
              <div class="small-note">Indicative range: ${lower:,.0f} – ${upper:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("The ±10% range is a simple presentation range, not a formal statistical confidence interval.")

with right:
    st.subheader("Model information")
    st.metric("Test R² score", f"{metadata['r2_score']:.3f}")
    st.metric("Training rows", f"{metadata['training_rows']:,}")
    st.markdown("**Algorithm:** Random Forest Regressor")
    st.markdown("**Target:** Diamond price in the dataset's currency (USD)")
    st.info(
        "This model was trained using the preprocessing and category ordering from your uploaded notebook. "
        "Predictions are most reliable for inputs similar to the training dataset."
    )

    with st.expander("Quality grade meanings"):
        st.markdown(
            """
            **Cut:** Fair → Good → Very Good → Premium → Ideal  
            **Color:** J (more color) → D (colorless)  
            **Clarity:** I1 → SI2 → SI1 → VS2 → VS1 → VVS2 → VVS1 → IF
            """
        )

st.divider()
st.caption("Built with Streamlit and scikit-learn.")
