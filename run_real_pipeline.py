"""
CUSTOMER ANALYTICS & CLV PREDICTION -- RUN ON REAL DATA
==========================================================
Dataset: UCI "Online Retail" -- 541,909 real transactions from a UK-based
online retailer, Dec 2010-Dec 2011. (Chen, Sain & Guo, 2012, J. Database
Marketing & Customer Strategy Mgmt.)

Methodology upgrade vs. the synthetic demo: since we have a real 373-day
timeline, we do a genuine time-based split instead of a simulated target --
  - Feature window : 2010-12-01 -> 2011-09-09  (~9 months, cutoff)
  - Target window   : 2011-09-09 -> 2011-12-09 (~3 months of REAL future spend)
CLV target = actual revenue each customer generated in the target window
(0 if they didn't return) -- this is real, out-of-time validation.
"""
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------------------------------------------------------------------
# STEP 1: LOAD + CLEAN REAL DATA
# ---------------------------------------------------------------------
raw = pd.read_csv("real_online_retail.csv")
raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
raw["OrderValue"] = raw["Quantity"] * raw["UnitPrice"]
raw["CustomerID"] = raw["CustomerID"].astype(str).str.replace(r"\.0$", "", regex=True)

# drop guest/unknown customers (CustomerID==0) and non-positive revenue lines (returns/adjustments)
raw = raw[(raw["CustomerID"] != "0") & (raw["OrderValue"] > 0)]

# collapse line items into one row per (customer, calendar day) = one "order"
orders = raw.groupby(["CustomerID", raw["InvoiceDate"].dt.date]).agg(
    OrderValue=("OrderValue", "sum"), Items=("Quantity", "sum")
).reset_index().rename(columns={"InvoiceDate": "OrderDate"})
orders["OrderDate"] = pd.to_datetime(orders["OrderDate"])

print(f"Cleaned real transactions: {len(raw):,} order lines -> {len(orders):,} customer-orders, "
      f"{orders.CustomerID.nunique():,} unique customers")
print(f"Date range: {orders.OrderDate.min().date()} to {orders.OrderDate.max().date()}")

# ---------------------------------------------------------------------
# STEP 2: TIME-BASED SPLIT + RFM FEATURES (from feature window only)
# ---------------------------------------------------------------------
CUTOFF_DATE = pd.Timestamp("2011-09-09")   # ~9 months in
END_DATE = orders.OrderDate.max() + timedelta(days=1)

feat_window = orders[orders.OrderDate < CUTOFF_DATE]
future_window = orders[orders.OrderDate >= CUTOFF_DATE]

rfm = feat_window.groupby("CustomerID").agg(
    Recency=("OrderDate", lambda x: (CUTOFF_DATE - x.max()).days),
    Frequency=("OrderDate", "count"),
    Monetary=("OrderValue", "sum"),
    AvgOrderValue=("OrderValue", "mean"),
    FirstPurchase=("OrderDate", "min"),
    TotalItems=("Items", "sum"),
).reset_index()
rfm["TenureDays"] = (CUTOFF_DATE - rfm["FirstPurchase"]).dt.days
rfm["AvgDaysBetweenOrders"] = rfm["TenureDays"] / rfm["Frequency"].clip(lower=1)
rfm = rfm.drop(columns=["FirstPurchase"])

# real forward-looking CLV: actual revenue this customer generated in the 3-month holdout
future_rev = future_window.groupby("CustomerID")["OrderValue"].sum().rename("CLV_Next3M")
rfm = rfm.merge(future_rev, on="CustomerID", how="left")
rfm["CLV_Next3M"] = rfm["CLV_Next3M"].fillna(0.0)

print(f"\nCustomers with >=1 order before cutoff (feature set): {len(rfm):,}")
print(f"Of those, % who returned & spent in the 3-month holdout: "
      f"{(rfm['CLV_Next3M']>0).mean()*100:.1f}%")
rfm.to_csv("rfm_features_real.csv", index=False)

# ---------------------------------------------------------------------
# STEP 3: K-MEANS CLUSTERING + PCA
# ---------------------------------------------------------------------
features = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "TenureDays"]
X = rfm[features].copy()
for col in ["Frequency", "Monetary", "AvgOrderValue"]:
    X[col] = np.log1p(X[col])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias, sils = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(list(K_range), inertias, marker="o")
axes[0].set_title("Elbow Method (real data)"); axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
axes[1].plot(list(K_range), sils, marker="o", color="darkorange")
axes[1].set_title("Silhouette Score (real data)"); axes[1].set_xlabel("k"); axes[1].set_ylabel("Score")
plt.tight_layout(); plt.savefig("plot_elbow_silhouette_REAL.png", dpi=150); plt.close()
print("\nSilhouette scores by k:", dict(zip(K_range, np.round(sils, 3))))

K_FINAL = 4
kmeans = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)
print(f"Final silhouette score (k={K_FINAL}): {silhouette_score(X_scaled, rfm['Cluster']):.3f}")

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
rfm["PC1"], rfm["PC2"] = coords[:, 0], coords[:, 1]
print("PCA explained variance:", np.round(pca.explained_variance_ratio_, 3),
      "-> total:", round(pca.explained_variance_ratio_.sum()*100, 1), "%")

plt.figure(figsize=(7, 6))
palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for c in sorted(rfm["Cluster"].unique()):
    sub = rfm[rfm["Cluster"] == c]
    plt.scatter(sub["PC1"], sub["PC2"], s=10, alpha=0.5, color=palette[c], label=f"Cluster {c}")
plt.title("Real Customer Segments (PCA-reduced RFM space)\nUCI Online Retail dataset")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.legend(); plt.tight_layout()
plt.savefig("plot_pca_clusters_REAL.png", dpi=150); plt.close()

profile = rfm.groupby("Cluster")[features].mean().round(1)
profile["CustomerCount"] = rfm["Cluster"].value_counts().sort_index()
print("\nCluster profile (real data):\n", profile)
rfm.to_csv("rfm_clustered_real.csv", index=False)

# ---------------------------------------------------------------------
# STEP 5: XGBOOST CLV MODEL (predicting REAL held-out 3-month spend)
# ---------------------------------------------------------------------
feature_cols = ["Recency", "Frequency", "Monetary", "AvgOrderValue",
                 "TenureDays", "AvgDaysBetweenOrders", "TotalItems"]
Xf = rfm[feature_cols].copy()
Xf = pd.concat([Xf, pd.get_dummies(rfm["Cluster"], prefix="Cluster")], axis=1)
y = rfm["CLV_Next3M"]

X_train, X_test, y_train, y_test = train_test_split(Xf, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

pred_train, pred_test = model.predict(X_train), model.predict(X_test)
r2_train, r2_test = r2_score(y_train, pred_train), r2_score(y_test, pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
mae_test = mean_absolute_error(y_test, pred_test)

print(f"\n=== XGBoost CLV Model on REAL data (predicting genuine 3-month held-out revenue) ===")
print(f"Train R^2: {r2_train:.4f}")
print(f"Test  R^2: {r2_test:.4f}")
print(f"Test RMSE: {rmse_test:.2f}")
print(f"Test MAE : {mae_test:.2f}")

importances = pd.Series(model.feature_importances_, index=Xf.columns).sort_values(ascending=False)
print("\nFeature importances:\n", importances.round(3))

plt.figure(figsize=(7, 5))
importances.sort_values().plot(kind="barh", color="#4C72B0")
plt.title("XGBoost Feature Importance -- Real Data CLV Model")
plt.tight_layout(); plt.savefig("plot_feature_importance_REAL.png", dpi=150); plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred_test, alpha=0.35, s=15, color="#55A868")
lims = [0, max(y_test.max(), pred_test.max())]
plt.plot(lims, lims, "r--", linewidth=1.5)
plt.xlabel("Actual 3-Month Future Revenue (£)")
plt.ylabel("Predicted 3-Month Future Revenue (£)")
plt.title(f"Actual vs Predicted CLV -- Real Data (Test R^2 = {r2_test:.3f})")
plt.tight_layout(); plt.savefig("plot_actual_vs_predicted_REAL.png", dpi=150); plt.close()

rfm["Predicted_CLV"] = model.predict(Xf)
rfm.to_csv("rfm_clv_predictions_real.csv", index=False)
model.save_model("xgb_clv_model_real.json")

# ---------------------------------------------------------------------
# STEP 6: BUSINESS VALUE
# ---------------------------------------------------------------------
profile2 = rfm.groupby("Cluster").agg(
    AvgRecency=("Recency", "mean"), AvgFrequency=("Frequency", "mean"),
    AvgMonetary=("Monetary", "mean"), AvgPredictedCLV=("Predicted_CLV", "mean"),
    CustomerCount=("CustomerID", "count"),
).round(1)
profile2["PctOfBase"] = (profile2["CustomerCount"] / profile2["CustomerCount"].sum() * 100).round(1)
profile2["TotalPredictedCLV"] = rfm.groupby("Cluster")["Predicted_CLV"].sum().round(0)
profile2["PctOfFutureRevenue"] = (profile2["TotalPredictedCLV"] / profile2["TotalPredictedCLV"].sum() * 100).round(1)

value_score = profile2["AvgFrequency"].rank() + profile2["AvgMonetary"].rank()
recency_rank = profile2["AvgRecency"].rank()
champion = value_score.idxmax()
rest = profile2.index.difference([champion])
newlow = value_score.loc[rest].idxmin()
rest2 = rest.difference([newlow])
atrisk = recency_rank.loc[rest2].idxmax()
loyal = rest2.difference([atrisk])[0]
label_map = {champion: "Champions / High-Value", newlow: "New / Low-Value",
             atrisk: "At-Risk / Churning", loyal: "Loyal / Core"}
profile2["SegmentName"] = profile2.index.map(label_map)
profile2 = profile2.sort_values("AvgPredictedCLV", ascending=False)
profile2.to_csv("segment_business_summary_real.csv")

print("\n=== SEGMENT BUSINESS SUMMARY (real data) ===")
print(profile2[["SegmentName", "CustomerCount", "PctOfBase", "AvgRecency",
                 "AvgFrequency", "AvgMonetary", "AvgPredictedCLV",
                 "TotalPredictedCLV", "PctOfFutureRevenue"]])

top = profile2.index[0]
print(f"\nHEADLINE: '{profile2.loc[top,'SegmentName']}' segment = "
      f"{profile2.loc[top,'PctOfBase']}% of customers but "
      f"{profile2.loc[top,'PctOfFutureRevenue']}% of predicted 3-month future revenue.")
print(f"Total predicted 3-month forward revenue across base: £{rfm['Predicted_CLV'].sum():,.0f}")

# ---------------------------------------------------------------------
# STEP 5b: REFINED MODEL -- log-transformed target (handles extreme skew
# from wholesale/whale customers -- standard practice for CLV data)
# ---------------------------------------------------------------------
y_log = np.log1p(rfm["CLV_Next3M"])
X_train2, X_test2, y_train2, y_test2 = train_test_split(Xf, y_log, test_size=0.2, random_state=42)

model_log = xgb.XGBRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
    random_state=42, n_jobs=-1,
)
model_log.fit(X_train2, y_train2, eval_set=[(X_test2, y_test2)], verbose=False)

pred_train2, pred_test2 = model_log.predict(X_train2), model_log.predict(X_test2)
r2_train_log = r2_score(y_train2, pred_train2)
r2_test_log = r2_score(y_test2, pred_test2)

y_test2_actual = np.expm1(y_test2)
pred_test2_actual = np.expm1(pred_test2).clip(min=0)
rmse_log_actual = np.sqrt(mean_squared_error(y_test2_actual, pred_test2_actual))
mae_log_actual = mean_absolute_error(y_test2_actual, pred_test2_actual)
r2_actual_scale = r2_score(y_test2_actual, pred_test2_actual)

print(f"\n=== REFINED MODEL: log1p(CLV) target (handles whale-customer skew) ===")
print(f"Log-space  Train R^2: {r2_train_log:.4f}")
print(f"Log-space  Test  R^2: {r2_test_log:.4f}  <- headline metric for this dataset")
print(f"Original-£ Test  R^2: {r2_actual_scale:.4f} (dominated by a few whale accounts)")
print(f"Original-£ RMSE: £{rmse_log_actual:,.2f}   MAE: £{mae_log_actual:,.2f}")

plt.figure(figsize=(6,6))
plt.scatter(y_test2, pred_test2, alpha=0.4, s=15, color="#4C72B0")
lims = [0, max(y_test2.max(), pred_test2.max())]
plt.plot(lims, lims, "r--", linewidth=1.5)
plt.xlabel("Actual log(1+3-Month Future Revenue)")
plt.ylabel("Predicted log(1+3-Month Future Revenue)")
plt.title(f"Log-Space Actual vs Predicted (Test R^2 = {r2_test_log:.3f})\nReal UCI Online Retail Data")
plt.tight_layout()
plt.savefig("plot_actual_vs_predicted_REAL_logspace.png", dpi=150)
plt.close()

model_log.save_model("xgb_clv_model_real_logspace.json")
