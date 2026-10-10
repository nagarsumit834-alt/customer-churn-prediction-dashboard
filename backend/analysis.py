import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data Loading
df=pd.read_csv("backend/customer_churn_dataset.csv")
df=df.head(10000)


#1) churn rate
churn_rate=df["Churn"].mean()*100
print("Churn rate=",churn_rate)

#2) Reteention Rate
retention_rate=(100-churn_rate)
print("Retention rate",retention_rate)

#3)Churn plan type
plan_type=df.groupby("Subscription Type")["Churn"].mean()*100
print(plan_type)

#4)ARPU - Average revenue per user
arpu=df["Total Spend"].mean()
print("Average spend of user =",arpu)

#5) Average tenure
avg_tenure=df["Tenure"].mean()
print("Avg_Tenure:",avg_tenure)

#6) Revenue loss due to churn user
Revenue_loss=df.loc[df["Churn"]==1,"Total Spend"].sum()
print("Revenue_loss(Lakh)",round(Revenue_loss/100000,2))

#7) Avg supportcall by churned user
avg_support_call=df.loc[df['Churn']==1,"Support Calls"].mean()
print("Avg Call",avg_support_call)

#8)Avg days of payment delay by churned users
avg_payment_delay=df.loc[df['Churn']==1,"Payment Delay"].mean()
print("Payment Delay by churned user",avg_payment_delay)

#9)churned rate based on Age
avg_churned_age=df.loc[df['Churn']==1,"Age"].mean()
print("Avg Age of churned user",avg_churned_age)

#10) Avg Last interaction of churned user
avg_last_interaction=df.loc[df['Churn']==1,"Last Interaction"].mean()
print("Avg last interaction of churned user:",avg_last_interaction)


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

os.makedirs("static", exist_ok=True)
df.head(10000).to_csv("backend/preprocessed_churn_data.csv", index=False)

# Data Loading
df = pd.read_csv("backend/customer_churn_dataset.csv").head(10000)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(15, 13))

# 1. Overall Churn vs Retention Rate
churn_counts = df["Churn"].value_counts()
axes[0, 0].pie(
    churn_counts,
    labels=["Retained", "Churned"],
    autopct="%1.1f%%",
    colors=["#2ecc71", "#e74c3c"],
    explode=(0, 0.08),
    startangle=140,
    shadow=True,
)
axes[0, 0].set_title(
    "1. Overall Churn vs Retention Rate", fontsize=12, fontweight="bold"
)

# 2. Churn Rate by Subscription Type
plan_type = (
    df.groupby("Subscription Type")["Churn"].mean() * 100
).reset_index()
sns.barplot(
    data=plan_type,
    x="Subscription Type",
    y="Churn",
    ax=axes[0, 1],
    palette="Purples_d",
)
axes[0, 1].set_title(
    "2. Churn Rate by Subscription Type (%)", fontsize=12, fontweight="bold"
)
axes[0, 1].set_ylabel("Churn Rate (%)")

# 3. Support Calls Line Plot (CORRECTED LINE PLOT)
support_calls_churn = df.groupby("Support Calls")["Churn"].mean() * 100
axes[1, 0].plot(
    support_calls_churn.index,
    support_calls_churn.values,
    color="red",
    marker="o",
    linestyle="dashed",
    linewidth=2,
)
axes[1, 0].set_title(
    "3. Churn Rate by Support Calls (%)", fontsize=12, fontweight="bold"
)
axes[1, 0].set_xlabel("Support Calls")
axes[1, 0].set_ylabel("Churn Rate (%)")
axes[1, 0].grid(True)

# 4. Payment Delay Density
sns.kdeplot(
    data=df,
    x="Payment Delay",
    hue="Churn",
    ax=axes[1, 1],
    fill=True,
    palette=["#2ecc71", "#e74c3c"],
    common_norm=False,
)
axes[1, 1].set_title(
    "4. Payment Delay Density (Days)", fontsize=12, fontweight="bold"
)

# 5. Revenue Impact
sns.barplot(
    data=df,
    x="Churn",
    y="Total Spend",
    ax=axes[2, 0],
    estimator=sum,
    palette=["#2ecc71", "#e74c3c"],
    errorbar=None,
)
axes[2, 0].set_title(
    "5. Total Revenue Impact ($)", fontsize=12, fontweight="bold"
)
axes[2, 0].set_xticklabels(["Retained Revenue", "Lost Revenue (Churn)"])

# 6. Age Distribution
sns.histplot(
    data=df,
    x="Age",
    hue="Churn",
    ax=axes[2, 1],
    kde=True,
    element="step",
    palette=["#2ecc71", "#e74c3c"],
)
axes[2, 1].set_title(
    "6. Age Distribution of Users", fontsize=12, fontweight="bold"
)

plt.tight_layout()
plt.savefig("static/churn_kpi_dashboard.png", dpi=300)
plt.show()
