import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ===============================
# Load model & data
# ===============================
model = joblib.load("model_supermarket.pkl")
metrics = joblib.load("metrics_regression.pkl")
encoders = joblib.load("encoders.pkl")
y_test, y_pred = joblib.load("regression_plot_data.pkl")

st.title("📈 Prediksi Sales Supermarket (Regresi Linear)")

st.markdown("""
Aplikasi ini menggunakan **Regresi Linear**  
untuk memprediksi **Total Sales** berdasarkan data transaksi.
""")

# ===============================
# Tampilkan hasil regresi
# ===============================
st.subheader("📊 Hasil Evaluasi Regresi")

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{metrics['MAE']:.2f}")
col2.metric("MSE", f"{metrics['MSE']:.2f}")
col3.metric("R² Score", f"{metrics['R2']:.2f}")

# ===============================
# Grafik regresi
# ===============================
st.subheader("📉 Grafik Actual vs Predicted Sales")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Sales")
ax.set_ylabel("Predicted Sales")
ax.set_title("Actual vs Predicted Sales")
st.pyplot(fig)

# ===============================
# Input User
# ===============================
st.subheader("🧾 Input Data Transaksi")

branch = st.selectbox("Branch", encoders["Branch"].classes_)
city = st.selectbox("City", encoders["City"].classes_)
customer_type = st.selectbox("Customer Type", encoders["Customer type"].classes_)
gender = st.selectbox("Gender", encoders["Gender"].classes_)
product_line = st.selectbox("Product Line", encoders["Product line"].classes_)
payment = st.selectbox("Payment Method", encoders["Payment"].classes_)

unit_price = st.number_input("Unit Price", min_value=0.0)
quantity = st.number_input("Quantity", min_value=1)
tax = st.number_input("Tax 5%", min_value=0.0)
cogs = st.number_input("COGS", min_value=0.0)
gross_income = st.number_input("Gross Income", min_value=0.0)
rating = st.slider("Rating", 1.0, 10.0)

# ===============================
# Encode input
# ===============================
input_data = pd.DataFrame([{
    "Branch": encoders["Branch"].transform([branch])[0],
    "City": encoders["City"].transform([city])[0],
    "Customer type": encoders["Customer type"].transform([customer_type])[0],
    "Gender": encoders["Gender"].transform([gender])[0],
    "Product line": encoders["Product line"].transform([product_line])[0],
    "Unit price": unit_price,
    "Quantity": quantity,
    "Tax 5%": tax,
    "Payment": encoders["Payment"].transform([payment])[0],
    "cogs": cogs,
    "gross income": gross_income,
    "Rating": rating
}])

# ===============================
# Prediksi
# ===============================
if st.button("🔮 Prediksi Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Prediksi Total Sales: {prediction:.2f}")
