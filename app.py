
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="CarPrice | Used Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# Load trained model
model = joblib.load("used_car_price_model.pkl")

# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(120deg, #14213d, #315b91);
    padding: 30px;
    border-radius: 15px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 38px;
}

.hero p {
    color: #e0e8f5;
    font-size: 17px;
}

.result {
    background: #e8f5ee;
    border-left: 6px solid #16834a;
    padding: 25px;
    border-radius: 12px;
    margin-top: 25px;
}

.result h2 {
    color: #126b3d;
    font-size: 34px;
}

.footer {
    text-align: center;
    color: gray;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)


# Header
st.markdown("""
<div class="hero">
    <h1>🚗 CarPrice</h1>
    <p>Smart Used Car Price Prediction System</p>
    <p>Estimate your car's market value using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)


# Brand and category options from training dataset
brands = [
    'Ambassador', 'Audi', 'BMW', 'Chevrolet', 'Daewoo',
    'Datsun', 'Fiat', 'Force', 'Ford', 'Honda', 'Hyundai',
    'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 'MG',
    'Mahindra', 'Maruti', 'Mercedes-Benz', 'Mitsubishi',
    'Nissan', 'OpelCorsa', 'Renault', 'Skoda', 'Tata',
    'Toyota', 'Volkswagen', 'Volvo'
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


# Input section
st.subheader("Enter Your Car Details")
st.write("Provide the following information to estimate your car price.")

with st.form("car_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Car Brand", brands)

        fuel = st.selectbox("Fuel Type", fuel_types)

        seller_type = st.selectbox("Seller Type", seller_types)

        transmission = st.selectbox(
            "Transmission",
            transmission_types
        )

    with col2:
        owner = st.selectbox("Owner Type", owner_types)

        car_age = st.number_input(
            "Car Age (Years)",
            min_value=0,
            max_value=40,
            value=5
        )

        km_driven = st.number_input(
            "Kilometres Driven",
            min_value=0,
            max_value=1000000,
            value=30000,
            step=1000
        )

    submitted = st.form_submit_button(
        "Calculate Estimated Price",
        use_container_width=True
    )


# Prediction
if submitted:

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
        <h3>Estimated Car Selling Price</h3>
        <h2>₹ {prediction:,.0f}</h2>
        <p>Your estimated used car value based on the selected details.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Note: This is a machine learning-based estimate. "
        "Actual selling prices may vary depending on vehicle "
        "condition, location, maintenance and market demand."
    )


# Footer
st.markdown("""
<div class="footer">
    Developed using Python, Scikit-learn and Streamlit
    <br>
    Used Car Price Prediction Project
</div>
""", unsafe_allow_html=True)
