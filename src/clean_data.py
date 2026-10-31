
import pandas as pd
import numpy as np

# 1. Load the original dataset
df = pd.read_csv("data/amazon_delivery.csv")

print("Original dataset shape:", df.shape)

# 2. Convert text placeholders such as "NaN" into real missing values
df = df.replace(r"^\s*NaN\s*$", np.nan, regex=True)

# 3. Remove exact duplicate rows, if any
duplicates = df.duplicated().sum()
df = df.drop_duplicates()
print("Duplicate rows removed:", duplicates)

# 4. Correct invalid agent ratings
# Ratings are expected to be between 1 and 5
invalid_ratings = (
    (df["Agent_Rating"] < 1)
    | (df["Agent_Rating"] > 5)
)
print("Invalid ratings found:", invalid_ratings.sum())

df.loc[invalid_ratings, "Agent_Rating"] = np.nan

# 5. Fill missing agent ratings with the median
rating_median = df["Agent_Rating"].median()
df["Agent_Rating"] = df["Agent_Rating"].fillna(rating_median)

# 6. Handle missing categorical and time values
df["Weather"] = df["Weather"].fillna("Unknown")
df["Traffic"] = df["Traffic"].fillna("Unknown")
df["Order_Time"] = df["Order_Time"].fillna("Unknown")

# 7. Convert date and time fields to appropriate formats
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"], errors="coerce"
)

df["Pickup_Time"] = pd.to_datetime(
    df["Pickup_Time"], format="%H:%M:%S", errors="coerce"
).dt.time

# 8. Calculate straight-line distance using the Haversine formula
lat1 = np.radians(df["Store_Latitude"])
lon1 = np.radians(df["Store_Longitude"])
lat2 = np.radians(df["Drop_Latitude"])
lon2 = np.radians(df["Drop_Longitude"])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
)

a = np.clip(a, 0, 1)

df["Distance_km"] = (
    2 * 6371 * np.arcsin(np.sqrt(a))
)

# 9. Flag suspicious distances instead of deleting records
df["Suspicious_Distance"] = df["Distance_km"] > 100

suspicious_count = df["Suspicious_Distance"].sum()
print("Distances above 100 km:", suspicious_count)

# Do not use implausible distances as valid distance measurements
df.loc[df["Suspicious_Distance"], "Distance_km"] = np.nan

# 10. Create useful time features
# These can help with later analysis and prediction
df["Order_Hour"] = pd.to_datetime(
    df["Order_Time"], format="%H:%M:%S", errors="coerce"
).dt.hour

df["Order_DayOfWeek"] = df["Order_Date"].dt.dayofweek

# 11. Save the cleaned dataset separately
output_path = "data/amazon_delivery_cleaned.csv"
df.to_csv(output_path, index=False)

# 12. Display a cleaning summary
print("\nCLEANING SUMMARY")
print("Final dataset shape:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDelivery time summary (minutes):")
print(df["Delivery_Time"].describe())

print("\nCleaned dataset saved to:", output_path)
