import streamlit as st
import pandas as pd
import joblib

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("model_supermarket.pkl")

st.set_page_config(page_title="Supermarket Classification", layout="centered")

st.title("📊 Klasifikasi Customer Type Supermarket")
st.write("Aplikasi ini memprediksi **Customer Type (Member / Normal)**")

st.divider()

# ==============================
# INPUT USER
# ==============================
st.subheader("🧾 Input Data Transaksi")

branch = st.selectbox("Branch", ["A", "B", "C"])
city = st.selectbox("City", ["Yangon", "Mandalay", "Naypyitaw"])
customer_gender = st.selectbox("Gender", ["Male", "Female"])
product_line = st.selectbox(
    "Product Line",
    [
        "Health and beauty",
        "Electronic accessories",
        "Home and lifestyle",
        "Sports and travel",
        "Food and beverages",
        "Fashion accessories"
    ]
)
payment = st.selectbox("Payment Method", ["Cash", "Credit card", "Ewallet"])

unit_price = st.number_input("Unit Price", min_value=0.0)
quantity = st.number_input("Quantity", min_value=1, step=1)
tax = st.number_input("Tax 5%", min_value=0.0)
sales = st.number_input("Sales", min_value=0.0)
cogs = st.number_input("COGS", min_value=0.0)
gross_income = st.number_input("Gross Income", min_value=0.0)
rating = st.slider("Rating", 1.0, 10.0)

# ==============================
# MANUAL ENCODING (HARUS SAMA DENGAN TRAINING)
# ==============================
branch_map = {"A": 0, "B": 1, "C": 2}
city_map = {"Yangon": 0, "Mandalay": 1, "Naypyitaw": 2}
gender_map = {"Male": 1, "Female": 0}
product_map = {
    "Health and beauty": 0,
    "Electronic accessories": 1,
    "Home and lifestyle": 2,
    "Sports and travel": 3,
    "Food and beverages": 4,
    "Fashion accessories": 5
}
payment_map = {"Cash": 0, "Credit card": 1, "Ewallet": 2}

# ==============================
# PREDIKSI
# ==============================
if st.button("🔍 Prediksi Customer Type"):
    input_data = pd.DataFrame([[
        branch_map[branch],
        city_map[city],
        customer_gender == "Male",
        product_map[product_line],
        unit_price,
        quantity,
        tax,
        sales,
        payment_map[payment],
        cogs,
        gross_income,
        rating
    ]], columns=[
        'Branch', 'City', 'Gender', 'Product line',
        'Unit price', 'Quantity', 'Tax 5%', 'Sales',
        'Payment', 'cogs', 'gross income', 'Rating'
    ])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Prediksi: **MEMBER**")
    else:
        st.warning("ℹ️ Prediksi: **NORMAL**")
