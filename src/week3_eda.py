
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("data/amazon_delivery_cleaned.csv")

# Display dataset information
print("Dataset Shape:", df.shape)

# Calculate delivery time statistics
delivery_time = df["Delivery_Time"]

print("\nDELIVERY TIME STATISTICS")
print("Mean:", delivery_time.mean())
print("Median:", delivery_time.median())
print("Mode:", delivery_time.mode()[0])
print("Minimum:", delivery_time.min())
print("Maximum:", delivery_time.max())
print("Range:", delivery_time.max() - delivery_time.min())
print("Standard Deviation:", delivery_time.std())

print("\nQuartiles:")
print(delivery_time.quantile([0.25, 0.50, 0.75]))


import matplotlib.pyplot as plt
import seaborn as sns

# Create delivery time distribution histogram
plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="Delivery_Time",
    bins=25,
    kde=True
)

plt.title("Distribution of Delivery Times")
plt.xlabel("Delivery Time (Minutes)")
plt.ylabel("Number of Deliveries")

plt.tight_layout()

plt.savefig(
    "visualizations/delivery_time_distribution.png",
    dpi=300
)

plt.show()
# Analyse delivery time by traffic condition

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Traffic",
    y="Delivery_Time"
)

plt.title("Delivery Time by Traffic Condition")
plt.xlabel("Traffic Condition")
plt.ylabel("Delivery Time (Minutes)")

plt.tight_layout()

plt.savefig(
    "visualizations/traffic_vs_delivery_time.png",
    dpi=300
)

plt.show()

# Analyse delivery time by weather condition

plt.figure(figsize=(11, 6))

sns.boxplot(
    data=df,
    x="Weather",
    y="Delivery_Time"
)

plt.title("Delivery Time by Weather Condition")
plt.xlabel("Weather Condition")
plt.ylabel("Delivery Time (Minutes)")
plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "visualizations/weather_vs_delivery_time.png",
    dpi=300
)

plt.show()
# Analyse relationship between distance and delivery time

distance_df = df.dropna(subset=["Distance_km"])

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=distance_df,
    x="Distance_km",
    y="Delivery_Time",
    alpha=0.4
)

plt.title("Distance vs Delivery Time")
plt.xlabel("Delivery Distance (KM)")
plt.ylabel("Delivery Time (Minutes)")

plt.tight_layout()

plt.savefig(
    "visualizations/distance_vs_delivery_time.png",
    dpi=300
)

plt.show()

# Calculate correlation
correlation = distance_df["Distance_km"].corr(
    distance_df["Delivery_Time"]
)

print("\nDISTANCE AND DELIVERY TIME CORRELATION")
print("Correlation:", correlation)
# Analyse average delivery time by area

area_time = df.groupby("Area")["Delivery_Time"].mean().sort_values()

plt.figure(figsize=(9, 6))

area_time.plot(kind="bar")

plt.title("Average Delivery Time by Area")
plt.xlabel("Delivery Area")
plt.ylabel("Average Delivery Time (Minutes)")
plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "visualizations/area_vs_delivery_time.png",
    dpi=300
)

plt.show()
# FINAL CORRELATION ANALYSIS

numeric_columns = [
    "Delivery_Time",
    "Distance_km",
    "Agent_Age",
    "Agent_Rating",
    "Order_Hour",
    "Order_DayOfWeek"
]

correlation_matrix = df[numeric_columns].corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation_matrix)

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Numerical Variables")
plt.tight_layout()

plt.savefig(
    "visualizations/correlation_heatmap.png",
    dpi=300
)

plt.show()

print("\nAVERAGE DELIVERY TIME BY AREA")
print(area_time)
# Correlation analysis of numerical variables

numeric_columns = [
    "Delivery_Time",
    "Distance_km",
    "Agent_Age",
    "Agent_Rating",
    "Order_Hour",
    "Order_DayOfWeek"
]

correlation_matrix = df[numeric_columns].corr()

print("\nCORRELATION MATRIX")
print(correlation_matrix)

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Numerical Variables")

plt.tight_layout()

plt.savefig(
    "visualizations/correlation_heatmap.png",
    dpi=300
)

plt.show()
# SHIPMENT VOLUME ANALYSIS

# 1. Number of deliveries by area
area_volume = df["Area"].value_counts()

print("\nSHIPMENT VOLUME BY AREA")
print(area_volume)

plt.figure(figsize=(9, 5))

area_volume.plot(kind="bar")

plt.title("Shipment Volume by Delivery Area")
plt.xlabel("Delivery Area")
plt.ylabel("Number of Deliveries")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "visualizations/shipment_volume_by_area.png",
    dpi=300
)

plt.show()


# 2. Number of deliveries by product category
category_volume = df["Category"].value_counts()

print("\nSHIPMENT VOLUME BY PRODUCT CATEGORY")
print(category_volume)

plt.figure(figsize=(10, 6))

category_volume.plot(kind="bar")

plt.title("Shipment Volume by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Number of Deliveries")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    "visualizations/shipment_volume_by_category.png",
    dpi=300
)

plt.show()