# WEEK 4: DELIVERY TIME PREDICTION

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the cleaned dataset
df = pd.read_csv("data/amazon_delivery_cleaned.csv")

# 2. Select input features and target
numeric_features = [
    "Distance_km",
    "Agent_Age",
    "Agent_Rating",
    "Order_Hour"
]

categorical_features = [
    "Weather",
    "Traffic",
    "Vehicle",
    "Area",
    "Category"
]

features = numeric_features + categorical_features

X = df[features]
y = df["Delivery_Time"]

# Remove rows where the target value is missing
valid_rows = y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 4. Prepare preprocessing steps
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 5. Create two prediction models
linear_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

forest_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# 6. Train both models
print("\nTraining Linear Regression...")
linear_model.fit(X_train, y_train)

print("Training Random Forest...")
forest_model.fit(X_train, y_train)

# 7. Evaluate model performance
def evaluate_model(model, model_name):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"\n{model_name} Results")
    print("MAE:", round(mae, 2), "minutes")
    print("RMSE:", round(rmse, 2), "minutes")
    print("R2 Score:", round(r2, 3))

    return predictions, mae, rmse, r2


linear_predictions, linear_mae, linear_rmse, linear_r2 = evaluate_model(
    linear_model, "Linear Regression"
)

forest_predictions, forest_mae, forest_rmse, forest_r2 = evaluate_model(
    forest_model, "Random Forest"
)

# 8. Compare models
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [linear_mae, forest_mae],
    "RMSE": [linear_rmse, forest_rmse],
    "R2 Score": [linear_r2, forest_r2]
})

print("\nMODEL COMPARISON")
print(results)

results.to_csv("Reports/week4_model_results.csv", index=False)

# 9. Cross-validation for Random Forest
print("\nRunning 3-fold cross-validation...")

cv_scores = cross_val_score(
    forest_model,
    X_train,
    y_train,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=1
)

cv_mae = -cv_scores

print("Cross-validation MAE scores:", cv_mae.round(2))
print("Average cross-validation MAE:", round(cv_mae.mean(), 2))

# 10. Actual vs predicted graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, forest_predictions, alpha=0.4)

minimum = min(y_test.min(), forest_predictions.min())
maximum = max(y_test.max(), forest_predictions.max())

plt.plot([minimum, maximum], [minimum, maximum], "r--")
plt.xlabel("Actual Delivery Time (minutes)")
plt.ylabel("Predicted Delivery Time (minutes)")
plt.title("Actual vs Predicted Delivery Time - Random Forest")
plt.tight_layout()
plt.savefig("visualizations/actual_vs_predicted.png")
plt.close()

# 11. Feature importance
trained_preprocessor = forest_model.named_steps["preprocessor"]
trained_forest = forest_model.named_steps["model"]

feature_names = trained_preprocessor.get_feature_names_out()
importance = trained_forest.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance", ascending=False
).head(15)

print("\nTOP 15 IMPORTANT FEATURES")
print(importance_df)

plt.figure(figsize=(10, 7))
sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)
plt.title("Top 15 Features Influencing Model Predictions")
plt.tight_layout()
plt.savefig("visualizations/feature_importance.png")
plt.close()

print("\nWeek 4 model execution completed!")
print("Graphs saved in the visualizations folder.")