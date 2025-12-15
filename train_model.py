# ===============================
# 1. IMPORT LIBRARY
# ===============================
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ===============================
# 2. LOAD DATASET
# ===============================
df = pd.read_csv("superMarket.csv")

print("Dataset loaded")
print(df.head())


# ===============================
# 3. PILIH FITUR & TARGET
# ===============================
FEATURES = [
    'Branch',
    'City',
    'Gender',
    'Product line',
    'Unit price',
    'Quantity',
    'Payment',
    'cogs',
    'gross income',
    'Rating'
]

TARGET = 'Sales'

X = df[FEATURES]
y = df[TARGET]


# ===============================
# 4. ENCODING DATA KATEGORIKAL
# ===============================
categorical_cols = [
    'Branch',
    'City',
    'Gender',
    'Product line',
    'Payment'
]

encoder = LabelEncoder()

for col in categorical_cols:
    X[col] = encoder.fit_transform(X[col])


# ===============================
# 5. SPLIT DATA TRAIN & TEST
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===============================
# 6. TRAIN MODEL REGRESI
# ===============================
model = LinearRegression()
model.fit(X_train, y_train)


# ===============================
# 7. EVALUASI MODEL
# ===============================
y_pred = model.predict(X_test)

print("\n=== Evaluasi Model Regresi ===")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))


# ===============================
# 8. VISUALISASI HASIL REGRESI
# ===============================
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()


# ===============================
# 9. SIMPAN MODEL & ENCODER
# ===============================
joblib.dump(model, "sales_regression_model.pkl")
joblib.dump(encoder, "encoder.pkl")

print("\nModel regresi dan encoder berhasil disimpan")
