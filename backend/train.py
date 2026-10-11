import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Load dataset
df = pd.read_csv("backend/customer_churn_dataset.csv", nrows=10000)

# 2. Define Features (X) and Target (y)
X = df.drop(columns=["CustomerID", "Churn","Gender"])
y = df["Churn"]

categorical_cols = [ "Subscription Type", "Contract Length"]

# 3. Create Preprocessing & Model Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough",
)

# Tuned Random Forest with stronger penalty for missing churners
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                class_weight={0: 1.0, 1: 2.5},  # Custom weight to boost churn prediction sensitivity
                random_state=42
            ),
        ),
    ]
)

# 4. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train Model
print("Training tuned Random Forest model...")
pipeline.fit(X_train, y_train)

# 6. Evaluate Performance
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n=== Model Evaluation Results ===")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}\n")
print(classification_report(y_test, y_pred))

# 7. Save Pipeline as joblib file
joblib.dump(pipeline, "backend/churn_model.joblib")
print("✓ Success! Model pipeline saved to 'backend/churn_model.joblib'")