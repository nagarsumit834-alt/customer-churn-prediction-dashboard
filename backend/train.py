import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Load dataset (10,000 rows sample)
df = pd.read_csv("backend/customer_churn_dataset.csv", nrows=10000)

# 2. Define Features (X) and Target (y)
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

categorical_cols = ["Gender", "Subscription Type", "Contract Length"]

# 3. Create Preprocessing & Model Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough",
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced"
            ),
        ),
    ]
)

# 4. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train Model
print("Training Random Forest model...")
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