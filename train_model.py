import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("superMarket.csv")
print("Dataset loaded")

# ===============================
# Drop kolom yang tidak dipakai
# ===============================
df = df.drop(columns=[
    "Invoice ID",
    "Date",
    "Time",
    "gross margin percentage"
])

# ===============================
# Encoding data kategorikal
# ===============================
label_cols = ["Branch", "City", "Customer type", "Gender", "Product line", "Payment"]

encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ===============================
# Fitur & Target
# ===============================
X = df.drop("Sales", axis=1)
y = df["Sales"]

# ===============================
# Split data
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Model Regresi Linear
# ===============================
model = LinearRegression()
model.fit(X_train, y_train)

# ===============================
# Prediksi & Evaluasi
# ===============================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("MAE :", mae)
print("MSE :", mse)
print("R2  :", r2)

# ===============================
# Simpan Model & Evaluasi
# ===============================
joblib.dump(model, "model_supermarket.pkl")
joblib.dump(encoders, "encoders.pkl")

metrics = {
    "MAE": mae,
    "MSE": mse,
    "R2": r2
}
joblib.dump(metrics, "metrics_regression.pkl")
joblib.dump((y_test, y_pred), "regression_plot_data.pkl")

print("Model, metrics, dan data plot berhasil disimpan")

# ===============================
# Grafik (opsional saat training)
# ===============================
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
