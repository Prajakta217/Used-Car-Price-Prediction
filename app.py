
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="CarValue | Used Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("used_car_price_model.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.hero {
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(120deg, #14213d, #315c9b);
    color: white;
    text-align: center;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 42px;
}
.hero p {
    color: #e5eaf5;
    font-size: 17px;
}
.result {
    background-color: #e8f5e9;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    color: #1b5e20;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🚗 CarValue</h1>
    <p>Smart Used Car Price Prediction System</p>
    <p>Discover the estimated market value of your car using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Price Prediction",
    "📊 Model Comparison",
    "⭐ Feature Importance",
    "ℹ️ About Project"
])

# ==================================================
# TAB 1: PRICE PREDICTION
# ==================================================
with tab1:

    st.subheader("Enter Car Details")
    st.write("Provide the following information to estimate your car's selling price.")

    brands = [
        'Ambassador', 'Audi', 'BMW', 'Chevrolet', 'Daewoo',
        'Datsun', 'Fiat', 'Force', 'Ford', 'Honda',
        'Hyundai', 'Isuzu', 'Jaguar', 'Jeep', 'Kia',
        'Land', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz',
        'Mitsubishi', 'Nissan', 'OpelCorsa', 'Renault',
        'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo'
    ]

    fuel_types = ['CNG', 'Diesel', 'Electric', 'LPG', 'Petrol']
    seller_types = ['Dealer', 'Individual', 'Trustmark Dealer']
    transmission_types = ['Automatic', 'Manual']

    owner_types = [
        'First Owner',
        'Fourth & Above Owner',
        'Second Owner',
        'Test Drive Car',
        'Third Owner'
    ]

    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Car Brand", brands)
        fuel = st.selectbox("Fuel Type", fuel_types)
        seller_type = st.selectbox("Seller Type", seller_types)
        transmission = st.selectbox(
            "Transmission", transmission_types
        )

    with col2:
        owner = st.selectbox("Owner Type", owner_types)
        car_age = st.number_input(
            "Car Age (Years)", min_value=0,
            max_value=40, value=5
        )
        km_driven = st.number_input(
            "Kilometres Driven", min_value=0,
            max_value=1000000, value=30000, step=1000
        )

    if st.button("Predict Car Price", type="primary", use_container_width=True):

        input_data = pd.DataFrame({
            'brand': [brand],
            'fuel': [fuel],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'owner': [owner],
            'car_age': [car_age],
            'km_driven': [km_driven]
        })

        prediction = model.predict(input_data)[0]

        st.markdown(f"""
        <div class="result">
            <h3>Estimated Selling Price</h3>
            <h1>₹ {prediction:,.0f}</h1>
            <p>Predicted using the Random Forest model</p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "This is an estimated price, not a guaranteed market offer."
        )

# ==================================================
# TAB 2: MODEL COMPARISON
# ==================================================
with tab2:

    st.subheader("Machine Learning Model Performance")

    comparison_df = pd.DataFrame({
        "Model": [
            "Random Forest",
            "Gradient Boosting",
            "Linear Regression",
            "Decision Tree"
        ],
        "MAE": [
            159204.467506,
            157754.250735,
            179642.899003,
            199731.959198
        ],
        "RMSE": [
            366340.635900,
            379171.773162,
            384722.646226,
            475334.798428
        ],
        "R2 Score": [
            0.583385,
            0.553690,
            0.540527,
            0.298604
        ]
    })

    st.dataframe(
        comparison_df.style.format({
            "MAE": "₹{:,.0f}",
            "RMSE": "₹{:,.0f}",
            "R2 Score": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### R² Score Comparison")
    st.bar_chart(
        comparison_df.set_index("Model")[["R2 Score"]]
    )

    st.markdown("### MAE Comparison")
    st.bar_chart(
        comparison_df.set_index("Model")[["MAE"]]
    )

    st.markdown("### RMSE Comparison")
    st.bar_chart(
        comparison_df.set_index("Model")[["RMSE"]]
    )

    st.info(
        "Random Forest was selected for deployment because it achieved "
        "the highest R² score and lowest RMSE among the four models. "
        "Gradient Boosting achieved a slightly lower MAE."
    )

# ==================================================
# TAB 3: FEATURE IMPORTANCE
# ==================================================
with tab3:

    st.subheader("What Influences Used Car Prices?")

    st.write(
        "Feature importance shows how much each input feature "
        "contributes to the Random Forest model's predictions."
    )

    importance_df = pd.DataFrame({
        "Feature": [
            "Car Age",
            "Kilometres Driven",
            "Automatic Transmission",
            "Manual Transmission",
            "Diesel",
            "Audi",
            "Land",
            "Toyota",
            "Petrol",
            "BMW",
            "Dealer",
            "Individual",
            "Maruti",
            "Mercedes-Benz",
            "Jaguar"
        ],
        "Importance": [
            0.224710,
            0.133043,
            0.126300,
            0.117726,
            0.078917,
            0.052505,
            0.040222,
            0.037116,
            0.033856,
            0.013928,
            0.011928,
            0.010997,
            0.010617,
            0.010368,
            0.010178
        ]
    })

    importance_df = importance_df.sort_values(
        by="Importance", ascending=True
    )

    st.markdown("### Top 15 Important Features")

    st.bar_chart(
        importance_df.set_index("Feature"),
        horizontal=True
    )

    st.caption(
        "Values represent the Random Forest model's feature importance "
        "for individual encoded features. They are not causal effects."
    )

    st.markdown("### Key Observations")

    st.write("""
    - Car age is the most important feature in this model.
    - Kilometres driven is another major factor.
    - Transmission type also contributes significantly.
    - Fuel type and car brand contribute to the predictions as well.
    """)

# ==================================================
# TAB 4: ABOUT PROJECT
# ==================================================
with tab4:

    st.subheader("About CarValue")

    st.write("""
    CarValue is a Machine Learning project designed to estimate
    the selling price of used cars based on their characteristics.
    """)

    st.markdown("### Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Original Records", "4,340")
    col2.metric("Cleaned Records", "3,577")
    col3.metric("Input Features", "7")

    st.markdown("### Features Used")

    st.write("""
    - Brand
    - Fuel Type
    - Seller Type
    - Transmission
    - Owner Type
    - Car Age
    - Kilometres Driven
    """)

    st.markdown("### Project Workflow")

    st.write("""
    1. Data collection
    2. Data cleaning and preprocessing
    3. Exploratory Data Analysis
    4. Feature engineering
    5. Model training and comparison
    6. Random Forest model selection
    7. Streamlit deployment
    """)

    st.markdown("### Technologies Used")

    st.write("""
    Python | Pandas | NumPy | Scikit-learn | Streamlit | Joblib
    """)

    st.markdown("### Project Links")

    st.markdown(
        "[GitHub Repository](https://github.com/Prajakta217/Used-Car-Price-Prediction)"
    )

    st.markdown(
        "[Live Application](https://used-car-price-prediction-prajakta.streamlit.app/)"
    )

# ---------------- FOOTER ----------------
st.divider()

st.markdown(
    "<center>🚗 CarValue | Machine Learning Based Used Car Price Prediction</center>",
    unsafe_allow_html=True
)
