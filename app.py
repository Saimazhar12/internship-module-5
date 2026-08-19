import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# --- Load model, scaler, feature list ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

model, scaler, features = load_artifacts()

st.title("🏠 House Price Predictor")
st.write(
    "Enter the details of a house below and get an instant AI-powered price estimate, "
    "using a Random Forest model (R² ≈ 0.99 on test data)."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
    sqft_living = st.number_input("Living Area (sqft)", min_value=200, max_value=15000, value=1800, step=50)
    sqft_lot = st.number_input("Lot Size (sqft)", min_value=500, max_value=50000, value=5000, step=100)

with col2:
    floors = st.selectbox("Floors", [1, 1.5, 2, 2.5, 3], index=1)
    age = st.number_input("Age of House (years)", min_value=0, max_value=150, value=15, step=1)
    garage = st.number_input("Garage Spaces", min_value=0, max_value=5, value=1, step=1)
    location_score = st.slider("Location Quality (1 = low, 10 = high)", 1.0, 10.0, 6.0, 0.1)

st.divider()

if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "age": age,
        "garage": garage,
        "location_score": location_score,
    }])[features]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.success(f"### Estimated Price: ${prediction:,.0f}")
    st.caption("This is an estimate based on a machine learning model trained on sample housing data. "
               "For a real deployment, retrain this model on your own local housing dataset for accurate results.")

st.divider()
with st.expander("ℹ️ About this app"):
    st.write(
        "- **Model**: Random Forest Regressor (500 trees)\n"
        "- **Preprocessing**: StandardScaler on all numeric features\n"
        "- **Note**: This demo was trained on a synthetically generated dataset "
        "(since the original dataset wasn't available). Swap in `house_price.csv` "
        "with your real data and re-run `train_model.py` to use your own numbers."
    )
