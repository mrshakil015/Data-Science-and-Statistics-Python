# ============================================
# Student Performance - ML Model
# ============================================

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# ============================================
# Task 1: Data Loading and Pre-processing
# ============================================

# Load dataset
df = pd.read_csv("student_performance.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Handle missing values
# For numeric columns -> fill with median
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# For categorical columns -> fill with mode
categorical_columns = df.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


# ============================================
# Create Target Variable
# ============================================

# Result = 1 if FinalScore >= 50
# Result = 0 if FinalScore < 50

df["Result"] = (df["FinalScore"] >= 50).astype(int)

print("\nTarget distribution:")
print(df["Result"].value_counts())


# ============================================
# Encode Categorical Columns
# ============================================

categorical_features = [
    "Gender",
    "ParentEducation",
    "InternetAccess"
]

df = pd.get_dummies(
    df,
    columns=categorical_features,
    drop_first=True
)

print("\nEncoded Data:")
print(df.head())


# ============================================
# Separate Features and Target
# ============================================

# FinalScore is removed because it was directly
# used to create Result.
X = df.drop(columns=["FinalScore", "Result"])

y = df["Result"]


# ============================================
# Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================
# Scale Numeric Features
# ============================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================
# Task 2: Model Training
# ============================================

model = LogisticRegression()

model.fit(X_train, y_train)

print("\nModel training completed.")


# ============================================
# Task 3: Model Evaluation
# ============================================

# Predictions
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Precision
precision = precision_score(y_test, y_pred)

# Recall
recall = recall_score(y_test, y_pred)


print("\n========== Model Evaluation ==========")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")


# ============================================
# Confusion Matrix
# ============================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================
# Task 4: Save the Model
# ============================================

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved as model.pkl")


# ============================================
# Reload the Model
# ============================================

with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

print("Model reloaded successfully.")


# ============================================
# Make One Test Prediction
# ============================================

sample = X_test[0].reshape(1, -1)

prediction = loaded_model.predict(sample)

print("\nTest Prediction:")

if prediction[0] == 1:
    print("Result: Pass")
else:
    print("Result: Fail")