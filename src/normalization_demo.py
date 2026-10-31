
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the cleaned dataset
df = pd.read_csv("data/amazon_delivery_cleaned.csv")

# Select two numerical features
features = df[["Agent_Age", "Distance_km"]].copy()

# Fill missing distances temporarily using the median
features["Distance_km"] = features["Distance_km"].fillna(
    features["Distance_km"].median()
)

# Apply Min-Max Scaling
minmax_scaler = MinMaxScaler()
minmax_data = minmax_scaler.fit_transform(features)

# Apply Standardization
standard_scaler = StandardScaler()
standard_data = standard_scaler.fit_transform(features)

print("Original feature values:")
print(features.head())

print("\nMin-Max Scaled values:")
print(pd.DataFrame(
    minmax_data,
    columns=["Agent_Age", "Distance_km"]
).head())

print("\nStandardized values:")
print(pd.DataFrame(
    standard_data,
    columns=["Agent_Age", "Distance_km"]
).head())

print("\nMin-Max range:")
print(pd.DataFrame(
    minmax_data,
    columns=["Agent_Age", "Distance_km"]
).agg(["min", "max"]))

print("\nStandardized mean and standard deviation:")
print(pd.DataFrame(
    standard_data,
    columns=["Agent_Age", "Distance_km"]
).agg(["mean", "std"]))
