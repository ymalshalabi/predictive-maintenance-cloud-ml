# Copyright (c) 2026 Yasmin Mazen AlShalabi 
# SPDX-License-Identifier: MIT

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('training_data.csv')

# Display dataset info
print(f"Dataset shape: {data.shape}")
print(f"Features: {list(data.columns)}")
print(f"Class distribution:\n{data['failure_within_1hr'].value_counts()}")
print("-" * 50)

# Preprocessing
print("Preprocessing data...")

# Convert timestamp to datetime and extract useful features
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['hour'] = data['timestamp'].dt.hour
data['minute'] = data['timestamp'].dt.minute
data['second'] = data['timestamp'].dt.second

# Encode categorical variable (server_type)
label_encoder = LabelEncoder()
data['server_type_encoded'] = label_encoder.fit_transform(data['server_type'])

# Define features and target
features = ['min_cpu', 'max_cpu', 'avg_cpu', 'memory_usage', 'disk_io',
            'network_latency', 'temperature', 'hour', 'minute', 'second',
            'server_type_encoded']

X = data[features]
y = data['failure_within_1hr']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print("-" * 50)

# ============================================
# Random Forest Training with 50 trees/epochs
# ============================================
print("\n" + "="*60)
print("TRAINING RANDOM FOREST CLASSIFIER")
print("="*60)

# Initialize Random Forest with warm_start to track progress
rf_model = RandomForestClassifier(
    n_estimators=50,  # 50 epochs/trees
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    warm_start=True,  # Allows incremental training
    verbose=0,
    n_jobs=-1
)

# Train incrementally to show progress
rf_model.n_estimators = 1
for epoch in range(1, 51):
    rf_model.n_estimators = epoch
    rf_model.fit(X_train, y_train)

    # Calculate training accuracy
    y_train_pred = rf_model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    # Calculate test accuracy
    y_test_pred = rf_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"RF Epoch {epoch:2d}/50 | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# Final evaluation
print("\nRandom Forest Final Evaluation:")
print(f"Final Training Accuracy: {accuracy_score(y_train, rf_model.predict(X_train)):.4f}")
print(f"Final Test Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_model.predict(X_test)))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_model.predict(X_test)))

# Save Random Forest model
rf_filename = 'random_forest_model.pkl'
with open(rf_filename, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"\nRandom Forest model saved as '{rf_filename}'")

# ============================================
# XGBoost Training with 50 epochs
# ============================================
print("\n" + "="*60)
print("TRAINING XGBOOST CLASSIFIER")
print("="*60)

# Prepare data for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters for binary classification
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 4,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'verbosity': 0
}

# Train with 50 boosting rounds
num_rounds = 50
evals_result = {}

print("Training XGBoost...")
model_xgb = xgb.train(
    params,
    dtrain,
    num_boost_round=num_rounds,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    evals_result=evals_result,
    verbose_eval=True  # This will print progress for each epoch
)

# Print final metrics
train_predictions = model_xgb.predict(dtrain)
qctions = model_xgb.predict(dtest)

# Convert probabilities to binary predictions
train_pred_binary = [1 if p > 0.5 else 0 for p in train_predictions]
test_pred_binary = [1 if p > 0.5 else 0 for p in test_predictions]

print(f"\nXGBoost Final Evaluation:")
print(f"Final Training Accuracy: {accuracy_score(y_train, train_pred_binary):.4f}")
print(f"Final Test Accuracy: {accuracy_score(y_test, test_pred_binary):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, test_pred_binary))
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred_binary))

# Save XGBoost model
xgb_filename = 'xgboost_model.pkl'
with open(xgb_filename, 'wb') as f:
    pickle.dump(model_xgb, f)
print(f"XGBoost model saved as '{xgb_filename}'")

# ============================================
# Additional: Feature Importance Analysis
# ============================================
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Random Forest Feature Importance
print("\nRandom Forest Feature Importance:")
rf_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(rf_importance.to_string(index=False))

# XGBoost Feature Importance - FIXED
print("\nXGBoost Feature Importance:")
# Get the importance scores from the model
importance_dict = model_xgb.get_score(importance_type='weight')

# Create a list to store importance values for all features
importance_values = []
for i in range(len(features)):
    # XGBoost uses 'f0', 'f1', etc. as feature names
    feature_key = f'f{i}'
    importance_values.append(importance_dict.get(feature_key, 0))

# Create DataFrame with feature names and importance values
xgb_importance = pd.DataFrame({
    'feature': features,
    'importance': importance_values
}).sort_values('importance', ascending=False)

print(xgb_importance.to_string(index=False))

# Alternative method using xgb.plot_importance
print("\nTop 5 Most Important Features (XGBoost):")
top_features = xgb_importance.head(5)
for idx, row in top_features.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# ============================================
# Model Comparison
# ============================================
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test))
xgb_test_acc = accuracy_score(y_test, test_pred_binary)

print(f"Random Forest Test Accuracy: {rf_test_acc:.4f}")
print(f"XGBoost Test Accuracy: {xgb_test_acc:.4f}")

if rf_test_acc > xgb_test_acc:
    print("Random Forest performed better!")
elif xgb_test_acc > rf_test_acc:
    print("XGBoost performed better!")
else:
    print("Both models performed equally!")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Models saved:")
print(f"1. '{rf_filename}' (Random Forest)")
print(f"2. '{xgb_filename}' (XGBoost)")
print("\nYou can load them later using:")
print("  with open('random_forest_model.pkl', 'rb') as f:")
print("      rf_model = pickle.load(f)")
print("  with open('xgboost_model.pkl', 'rb') as f:")
print("      xgb_model = pickle.load(f)")
