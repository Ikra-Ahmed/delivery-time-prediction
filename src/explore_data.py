import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("data/amazon_delivery.csv")

# Show first 5 rows
print("First 5 rows:")
print(df.head())

# Show column names
print("\nColumn names:")
print(df.columns.tolist())

# Show dataset information
print("\nDataset information:")
print(df.info())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())
print("\nUnique values:")

print("\nWeather:")
print(df["Weather"].unique())

print("\nTraffic:")
print(df["Traffic"].unique())

print("\nVehicle:")
print(df["Vehicle"].unique())

print("\nArea:")
print(df["Area"].unique())

print("\nCategory:")
print(df["Category"].unique())

print("\nDelivery Time statistics:")
print(df["Delivery_Time"].describe())

# Calculate approximate distance between store and drop location
lat1 = np.radians(df["Store_Latitude"])
lon1 = np.radians(df["Store_Longitude"])
lat2 = np.radians(df["Drop_Latitude"])
lon2 = np.radians(df["Drop_Longitude"])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
distance = 2 * 6371 * np.arcsin(np.sqrt(a))

print("\nCoordinate ranges:")
print("Store latitude:", df["Store_Latitude"].min(), "to", df["Store_Latitude"].max())
print("Store longitude:", df["Store_Longitude"].min(), "to", df["Store_Longitude"].max())
print("Drop latitude:", df["Drop_Latitude"].min(), "to", df["Drop_Latitude"].max())
print("Drop longitude:", df["Drop_Longitude"].min(), "to", df["Drop_Longitude"].max())

# Check suspicious coordinates
invalid_coordinates = (
    (df["Store_Latitude"] < 0) |
    (df["Store_Longitude"] < 0) |
    (df["Drop_Latitude"] < 0) |
    (df["Drop_Longitude"] < 0)
)

print("\nSuspicious coordinate records:")
print(invalid_coordinates.sum())

df["Distance_km"] = distance
print("\nDistance statistics:")
print(df["Distance_km"].describe())
# Traffic vs Delivery Time

print("\nAverage delivery time by traffic:")
print(df.groupby("Traffic")["Delivery_Time"].mean())
# Weather vs Delivery Time

print("\nAverage delivery time by weather:")
print(df.groupby("Weather")["Delivery_Time"].mean())

# Area vs Delivery Time

print("\nAverage delivery time by area:")
print(df.groupby("Area")["Delivery_Time"].mean())
print("\nNumber of deliveries by area:")
print(df["Area"].value_counts())
import matplotlib.pyplot as plt

# Traffic vs Delivery Time visualization

traffic_avg = df.groupby("Traffic")["Delivery_Time"].mean().dropna()

traffic_avg.plot(kind="bar")

plt.title("Average Delivery Time by Traffic")
plt.xlabel("Traffic")
plt.ylabel("Average Delivery Time (minutes)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("visualizations/traffic_delivery_time.png")
plt.show()
# Weather vs Delivery Time visualization

weather_avg = df.groupby("Weather")["Delivery_Time"].mean().dropna()

weather_avg.plot(kind="bar")

plt.title("Average Delivery Time by Weather")
plt.xlabel("Weather")
plt.ylabel("Average Delivery Time (minutes)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("visualizations/weather_delivery_time.png")
plt.show()
# Area vs Delivery Time visualization

area_avg = df.groupby("Area")["Delivery_Time"].mean().dropna()

area_avg.plot(kind="bar")

plt.title("Average Delivery Time by Area")
plt.xlabel("Area")
plt.ylabel("Average Delivery Time (minutes)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("visualizations/area_delivery_time.png")
plt.show()