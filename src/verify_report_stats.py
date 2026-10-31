
import pandas as pd
import numpy as np

# Load the original data
df = pd.read_csv("data/amazon_delivery.csv")

# Convert text placeholders into real missing values
df = df.replace(r"^\s*NaN\s*$", np.nan, regex=True)

# Check for negative coordinates
negative_coords = (
    (df["Store_Latitude"] < 0)
    | (df["Store_Longitude"] < 0)
    | (df["Drop_Latitude"] < 0)
    | (df["Drop_Longitude"] < 0)
)

# Calculate straight-line distance
lat1 = np.radians(df["Store_Latitude"])
lon1 = np.radians(df["Store_Longitude"])
lat2 = np.radians(df["Drop_Latitude"])
lon2 = np.radians(df["Drop_Longitude"])

a = (
    np.sin((lat2 - lat1) / 2) ** 2
    + np.cos(lat1) * np.cos(lat2)
    * np.sin((lon2 - lon1) / 2) ** 2
)

a = np.clip(a, 0, 1)
distance = 2 * 6371 * np.arcsin(np.sqrt(a))

long_distance = distance > 100

print("Rows with negative coordinates:", negative_coords.sum())
print("Rows with distance above 100 km:", long_distance.sum())
print("Rows meeting BOTH conditions:", (negative_coords & long_distance).sum())

# Check delivery-time outliers using the IQR method
q1 = df["Delivery_Time"].quantile(0.25)
q3 = df["Delivery_Time"].quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = (
    (df["Delivery_Time"] < lower)
    | (df["Delivery_Time"] > upper)
)

print("\nDelivery-time outlier analysis")
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower fence:", lower)
print("Upper fence:", upper)
print("Number of outliers:", outliers.sum())
