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
# 2. Add CORS Middleware (Explicit origins required when credentials are true)
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
  """Returns structured JSON data for 6 dynamic Chart.js visualizations."""
  if df is None:
    raise HTTPException(status_code=500, detail="Dataset not loaded")

  try:
    # 1. Subscription Type Churn
    sub_churn = (
        (df.groupby("Subscription Type")["Churn"].mean() * 100)
        .round(2)
        .to_dict()
    )

    # 2. Support Calls Churn
    calls_churn = (
        (df.groupby("Support Calls")["Churn"].mean() * 100).round(2).to_dict()
    )

    # 3. Contract Length Churn
    contract_churn = (
        (df.groupby("Contract Length")["Churn"].mean() * 100)
        .round(2)
        .to_dict()
    )

    # 4. Gender Churn
    gender_churn = (
        (df.groupby("Gender")["Churn"].mean() * 100).round(2).to_dict()
    )

    # 5. Tenure Groups Churn
    df_temp = df.copy()
    df_temp["Tenure_Group"] = pd.cut(
        df_temp["Tenure"],
        bins=[-1, 12, 24, 48, df_temp["Tenure"].max() + 1],
        labels=["0-1 Year", "1-2 Years", "2-4 Years", "4+ Years"],
    )
    tenure_churn = (
        (df_temp.groupby("Tenure_Group", observed=False)["Churn"].mean() * 100)
        .round(2)
        .to_dict()
    )

    # 6. Payment Delay Groups Churn
    df_temp["Delay_Group"] = pd.cut(
        df_temp["Payment Delay"],
        bins=[-1, 10, 20, 30, df_temp["Payment Delay"].max() + 1],
        labels=["0-10 Days", "11-20 Days", "21-30 Days", "30+ Days"],
    )
    delay_churn = (
        (df_temp.groupby("Delay_Group", observed=False)["Churn"].mean() * 100)
        .round(2)
        .to_dict()
    )

    return {
        "subscription_churn": {
            "labels": [str(k) for k in sub_churn.keys()],
            "data": [float(v) for v in sub_churn.values()],
        },
        "support_calls_churn": {
            "labels": [f"{k} Calls" for k in calls_churn.keys()],
            "data": [float(v) for v in calls_churn.values()],
        },
        "contract_churn": {
            "labels": [str(k) for k in contract_churn.keys()],
            "data": [float(v) for v in contract_churn.values()],
        },
        "gender_churn": {
            "labels": [str(k) for k in gender_churn.keys()],
            "data": [float(v) for v in gender_churn.values()],
        },
        "tenure_churn": {
            "labels": [str(k) for k in tenure_churn.keys()],
            "data": [0.0 if pd.isna(v) else float(v) for v in tenure_churn.values()],
        },
        "delay_churn": {
            "labels": [str(k) for k in delay_churn.keys()],
            "data": [0.0 if pd.isna(v) else float(v) for v in delay_churn.values()],
        },
    }
  except Exception as e:
    print(f"Error in breakdown endpoint: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))