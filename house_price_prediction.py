import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
housing = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target

# Save dataset as CSV (optional)
df.to_csv("house_prices.csv", index=False)

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
print("Model Performance")
print("-----------------")
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R² Score:", r2_score(y_test, predictions))

# Create charts folder if it doesn't exist
os.makedirs("charts", exist_ok=True)

# Scatter plot: Actual vs Predicted Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

# Save the chart
plt.savefig("charts/prediction_plot.png")
# Residual Plot (Prediction Errors)
plt.figure(figsize=(8, 6))
plt.scatter(predictions, y_test - predictions)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residual Error")
plt.title("Residual Plot")

plt.savefig("charts/residual_plot.png")

# Display the chart
plt.show()

# Save the trained model
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")