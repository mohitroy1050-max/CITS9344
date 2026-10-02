"""
House Price Prediction
Tech Stack: Python, Pandas, Scikit-learn, Matplotlib

This script loads housing data, trains a Linear Regression model,
evaluates it, and predicts prices for new houses.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
data = pd.read_csv("data/house_data.csv")
print("First 5 rows of dataset:")
print(data.head())

print("\nDataset info:")
print(data.describe())

# -------------------------------------------------
# 2. FEATURES AND TARGET
# -------------------------------------------------
X = data[["area_sqft", "bedrooms", "bathrooms", "age_years", "location_score"]]
y = data["price_lakhs"]

# -------------------------------------------------
# 3. TRAIN-TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# 4. TRAIN MODEL
# -------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------------------------
# 5. EVALUATE MODEL
# -------------------------------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error: {mae:.2f} lakhs")
print(f"Root Mean Squared Error: {rmse:.2f} lakhs")
print(f"R2 Score: {r2:.3f}")

print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.3f}")
print(f"  Intercept: {model.intercept_:.3f}")

# -------------------------------------------------
# 6. PREDICT PRICE FOR A NEW HOUSE (example)
# -------------------------------------------------
new_house = pd.DataFrame({
    "area_sqft": [1600],
    "bedrooms": [3],
    "bathrooms": [2],
    "age_years": [5],
    "location_score": [7]
})

predicted_price = model.predict(new_house)
print(f"\nPredicted price for sample new house: {predicted_price[0]:.2f} lakhs")

# -------------------------------------------------
# 7. VISUALIZATION: Actual vs Predicted
# -------------------------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, color="blue")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")
plt.xlabel("Actual Price (lakhs)")
plt.ylabel("Predicted Price (lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
print("\nChart saved as 'actual_vs_predicted.png'")
