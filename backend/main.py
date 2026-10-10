import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel, Field

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn probability and serving analytics.",
    version="1.0.0",
)

# 2. Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load Model and Dataset
try:
  model = joblib.load("backend/churn_model.joblib")
  print("✓ Successfully loaded backend/churn_model.joblib")
except Exception as e:
  print(f"Error loading model: {e}")
  model = None

try:
  df = pd.read_csv("backend/preprocessed_churn_data.csv")
  print("✓ Successfully loaded dataset preprocessed_churn_data.csv")
except Exception as e:
  print(f"Error loading dataset: {e}")
  df = None


# 4. Pydantic Input Schema
class CustomerFeatures(BaseModel):
  Age: int = Field(..., example=35)
  Gender: str = Field(..., example="Male")
  Tenure: int = Field(..., example=12)
  Usage_Frequency: int = Field(..., example=8)
  Support_Calls: int = Field(..., example=6)
  Payment_Delay: int = Field(..., example=25)
  Subscription_Type: str = Field(..., example="Basic")
  Contract_Length: str = Field(..., example="Monthly")
  Total_Spend: int = Field(..., example=300)
  Last_Interaction: int = Field(..., example=15)


# 5. Endpoints


@app.get("/")
def read_root():
  return {
      "message": "Welcome to the Churn Prediction API",
      "status": "Online",
      "docs_url": "/docs",
  }


@app.get("/api/analytics/summary")
def get_analytics_summary():
  """Returns expanded 8-metric KPI summary statistics."""
  if df is None:
    raise HTTPException(status_code=500, detail="Dataset not loaded")

  churn_df = df[df["Churn"] == 1]
  churn_rate = float(df["Churn"].mean() * 100)

  return {
      "total_customers": len(df),
      "churn_rate_pct": round(churn_rate, 2),
      "retention_rate_pct": round(100 - churn_rate, 2),
      "total_revenue": round(float(df["Total Spend"].sum()), 2),
      "avg_support_calls_churned": round(
          float(churn_df["Support Calls"].mean()), 2
      ),
      "avg_payment_delay_churned": round(
          float(churn_df["Payment Delay"].mean()), 2
      ),
      "avg_tenure_months": round(float(df["Tenure"].mean()), 1),
      "avg_last_interaction_days": round(
          float(churn_df["Last Interaction"].mean()), 1
      ),
  }


@app.get("/api/analytics/breakdown")
def get_analytics_breakdown():
  """Returns structured JSON data for dynamic Chart.js rendering."""
  if df is None:
    raise HTTPException(status_code=500, detail="Dataset not loaded")

  sub_churn = (
      (df.groupby("Subscription Type")["Churn"].mean() * 100)
      .round(2)
      .to_dict()
  )
  calls_churn = (
      (df.groupby("Support Calls")["Churn"].mean() * 100).round(2).to_dict()
  )

  return {
      "subscription_churn": {
          "labels": list(sub_churn.keys()),
          "data": list(sub_churn.values()),
      },
      "support_calls_churn": {
          "labels": [f"{k} Calls" for k in calls_churn.keys()],
          "data": list(calls_churn.values()),
      },
  }


@app.post("/api/predict")
def predict_churn(customer: CustomerFeatures):
  """Accepts customer details and returns churn prediction probability."""
  if model is None:
    raise HTTPException(status_code=500, detail="Model pipeline not loaded")

  input_data = pd.DataFrame([{
      "Age": customer.Age,
      "Gender": customer.Gender,
      "Tenure": customer.Tenure,
      "Usage Frequency": customer.Usage_Frequency,
      "Support Calls": customer.Support_Calls,
      "Payment Delay": customer.Payment_Delay,
      "Subscription Type": customer.Subscription_Type,
      "Contract Length": customer.Contract_Length,
      "Total Spend": customer.Total_Spend,
      "Last Interaction": customer.Last_Interaction,
  }])

  probability = float(model.predict_proba(input_data)[0][1])
  prediction = int(probability >= 0.5)

  return {
      "churn_prediction": prediction,
      "prediction_label": "Churn" if prediction == 1 else "Retained",
      "churn_probability": round(probability, 4),
      "risk_level": "High" if probability >= 0.5 else "Low",
  }