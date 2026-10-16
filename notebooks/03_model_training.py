# Databricks notebook source
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the feature-engineered dataset
df = spark.read.parquet("/Volumes/dbw_pred_maint_kkm/default/raw_data/features_train.parquet")
pdf = df.toPandas()

print(f"Dataset shape: {pdf.shape}")

# Split by engine_id — NOT random!
# Why? If we randomly split rows, the model could see cycle 50 and 52
# of the same engine in training, and predict cycle 51 in test.
# That's data leakage. In production, you predict on UNSEEN engines.
#
# We'll use engines 1-80 for training, 81-100 for validation.
train_engines = list(range(1, 81))
val_engines = list(range(81, 101))

train_df = pdf[pdf['engine_id'].isin(train_engines)].copy()
val_df = pdf[pdf['engine_id'].isin(val_engines)].copy()

# Define features (everything except engine_id, cycle, and RUL)
feature_cols = [c for c in pdf.columns if c not in ['engine_id', 'cycle', 'RUL']]
target_col = 'RUL'

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_val = val_df[feature_cols]
y_val = val_df[target_col]

print(f"Training set: {X_train.shape[0]} rows, {X_train.shape[1]} features")
print(f"Validation set: {X_val.shape[0]} rows, {X_val.shape[1]} features")
print(f"Training engines: {len(train_engines)}, Validation engines: {len(val_engines)}")

# COMMAND ----------

# Tree-based models (Random Forest, Gradient Boosting) don't need scaling,
# but Linear Regression does. We'll scale anyway since:
# 1. It doesn't hurt tree models
# 2. We want to compare fairly across model types
# 3. It's good practice for production pipelines

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_cols,
    index=X_train.index
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),  # Use transform, NOT fit_transform on validation!
    columns=feature_cols,
    index=X_val.index
)

print("Scaling complete")
print(f"Sample means after scaling (should be ~0): {X_train_scaled.mean().mean():.6f}")
print(f"Sample stds after scaling (should be ~1): {X_train_scaled.std().mean():.6f}")

# COMMAND ----------

# MLflow tracks every experiment: parameters, metrics, artifacts.
# This is what makes your project production-grade — you can compare
# models, reproduce results, and deploy the best one.

mlflow.set_experiment("/predictive-maintenance")

def evaluate_model(model, model_name, params, X_tr, y_tr, X_v, y_v):
    """Train a model and log everything to MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Train
        model.fit(X_tr, y_tr)

        # Predict
        y_train_pred = model.predict(X_tr)
        y_val_pred = model.predict(X_v)

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_tr, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_v, y_val_pred))
        val_mae = mean_absolute_error(y_v, y_val_pred)
        val_r2 = r2_score(y_v, y_val_pred)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_r2", val_r2)
        mlflow.sklearn.log_model(model, "model")

        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"  Train RMSE: {train_rmse:.2f}")
        print(f"  Val RMSE:   {val_rmse:.2f}")
        print(f"  Val MAE:    {val_mae:.2f}")
        print(f"  Val R²:     {val_r2:.4f}")

        return val_rmse

# --- Model 1: Linear Regression (baseline) ---
lr = LinearRegression()
lr_rmse = evaluate_model(
    lr, "Linear Regression",
    {"model_type": "linear_regression"},
    X_train_scaled, y_train, X_val_scaled, y_val
)

# --- Model 2: Random Forest ---
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_rmse = evaluate_model(
    rf, "Random Forest",
    {"model_type": "random_forest", "n_estimators": 100,
     "max_depth": 15, "min_samples_split": 5},
    X_train_scaled, y_train, X_val_scaled, y_val
)

# --- Model 3: Gradient Boosting ---
gb = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb_rmse = evaluate_model(
    gb, "Gradient Boosting",
    {"model_type": "gradient_boosting", "n_estimators": 200,
     "max_depth": 5, "learning_rate": 0.1},
    X_train_scaled, y_train, X_val_scaled, y_val
)

# COMMAND ----------

# Understanding WHICH sensors matter most is crucial for:
# 1. Explaining predictions to maintenance teams
# 2. Knowing which sensors to prioritize in monitoring
# 3. Interview discussions about model interpretability

import matplotlib.pyplot as plt

# Use the Random Forest or Gradient Boosting for feature importance
best_model = gb  # Usually gradient boosting performs best
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Top 15 features
top_n = 15
plt.figure(figsize=(12, 6))
plt.title("Top 15 Feature Importances (Gradient Boosting)")
plt.bar(range(top_n), importances[indices[:top_n]])
plt.xticks(range(top_n), [feature_cols[i] for i in indices[:top_n]], rotation=45, ha='right')
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("/Volumes/dbw_pred_maint_kkm/default/raw_data/feature_importance.png")
plt.show()

print("\nTop 15 features:")
for i in range(top_n):
    print(f"  {feature_cols[indices[i]]:35s}  {importances[indices[i]]:.4f}")

# COMMAND ----------

from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# Find the best run
experiment = mlflow.get_experiment_by_name("/predictive-maintenance")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
best_run = runs.sort_values("metrics.val_rmse").iloc[0]

print(f"Best model: {best_run['tags.mlflow.runName']}")
print(f"  Val RMSE: {best_run['metrics.val_rmse']:.2f}")
print(f"  Val MAE:  {best_run['metrics.val_mae']:.2f}")
print(f"  Val R²:   {best_run['metrics.val_r2']:.4f}")

# Re-log the best model WITH a signature
# The signature tells downstream systems exactly what input/output to expect
best_model_obj = rf  # Random Forest was best

signature = infer_signature(X_train_scaled, best_model_obj.predict(X_train_scaled))
print(f"\nModel signature:")
print(f"  Inputs:  {signature.inputs}")
print(f"  Outputs: {signature.outputs}")

with mlflow.start_run(run_name="Random Forest (with signature)"):
    mlflow.log_params({
        "model_type": "random_forest",
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5
    })
    mlflow.log_metric("val_rmse", float(best_run['metrics.val_rmse']))
    mlflow.log_metric("val_mae", float(best_run['metrics.val_mae']))
    mlflow.log_metric("val_r2", float(best_run['metrics.val_r2']))

    mlflow.sklearn.log_model(
        best_model_obj,
        "model",
        signature=signature,
        input_example=X_train_scaled.head(3)
    )

    run_id = mlflow.active_run().info.run_id

# Register in Unity Catalog
model_name = "dbw_pred_maint_kkm.default.predictive_maintenance_rul"
registered_model = mlflow.register_model(f"runs:/{run_id}/model", model_name)
print(f"\nRegistered model: {registered_model.name}")
print(f"Version: {registered_model.version}")

# COMMAND ----------

import pickle

# Save the scaler — we need it to transform incoming data the same way
scaler_path = "/Volumes/dbw_pred_maint_kkm/default/raw_data/scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {scaler_path}")

# Save feature columns list
import json
features_path = "/Volumes/dbw_pred_maint_kkm/default/raw_data/feature_cols.json"
with open(features_path, "w") as f:
    json.dump(feature_cols, f)
print(f"Feature columns saved to {features_path}")

# Save the trained model as pickle too (in addition to MLflow)
model_path = "/Volumes/dbw_pred_maint_kkm/default/raw_data/rf_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(rf, f)
print(f"Model saved to {model_path}")

print(f"\nFeature count: {len(feature_cols)}")
print(f"Features: {feature_cols}")