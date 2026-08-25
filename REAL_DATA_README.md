# Run on Real Data — UCI "Online Retail" Dataset

## What I used
Since you didn't have a file to upload, I pulled a genuine, well-known public
dataset: the **UCI Online Retail dataset** — 541,909 real transactions from
a UK-based online gift retailer, Dec 1, 2010 – Dec 9, 2011
(Chen, Sain & Guo, 2012, *J. Database Marketing & Customer Strategy Mgmt*).
Source: https://archive.ics.uci.edu/dataset/352/online+retail

After cleaning (dropped ~133K guest-checkout rows with no CustomerID, and
~1,200 return/cancellation lines with non-positive value): **397,884 real
order lines → 4,338 unique customers.**

`run_real_pipeline.py` reproduces everything below end-to-end.

## Key methodology change from the demo version
The synthetic demo used a *simulated* future-CLV target, which is why it hit
a clean 86% R². On real data I did something more rigorous and more
honest — a genuine **time-based split**:
- **Feature window:** Dec 1, 2010 → Sep 9, 2011 (RFM computed only from this)
- **Target window:** Sep 9, 2011 → Dec 9, 2011 (real, actually-observed spend)
- CLV target = **actual revenue each customer generated in that 3-month
  holdout** (£0 if they never came back)

This is out-of-time validation — the model never sees the future it's asked
to predict. It's a harder, fairer test than the synthetic version.

## Results — and why they're different from 86%

| | Raw £ scale | Log(1+£) scale |
|---|---|---|
| Test R² | **0.05** | **0.24** |
| Test RMSE | £6,643 | – |
| Test MAE | £771 | – |

![Actual vs predicted, log space](plot_actual_vs_predicted_REAL_logspace.png)

![Actual vs predicted, raw £ space](plot_actual_vs_predicted_REAL.png)

![Feature importance](plot_feature_importance_REAL.png)

**Why it's not 86% on real data:** customer spend here is extremely
right-skewed (skew = 22.5) — a small number of wholesale accounts spend
£10,000–£168,000 in three months while the median customer spends £176.
A handful of these "whale" customers dominate the squared-error term, so
raw-£ R² collapses even though the model's *relative* ranking of customers
is reasonably good. This is why CLV literature standardly evaluates on
**log(1+CLV)** — R² = 0.24 there is a realistic, credible result for
noisy, wholesaler-mixed retail data with only ~3,360 customers to learn
from and just 3 months of forward-looking signal.

**Takeaway:** an "86% R²" claim on a project like this usually means one of:
information leakage (e.g., using the same period's Monetary to predict
CLV defined over an overlapping period), a much larger/cleaner proprietary
dataset, a longer feature history, or a less noisy business (subscription
revenue is far easier to predict than one-off gift purchases). None of
that is dishonest to say to a stakeholder — it's normal, useful context for
interpreting the number.

## Clustering held up well on real data
- k=4 confirmed by silhouette scan (peak was actually k=3 at 0.325 vs 0.285
  at k=4 — worth knowing k=3 is a defensible alternative on this dataset)
- PCA (2 components, ~78% variance) shows real, if messier, cluster
  separation:

![PCA cluster plot](plot_pca_clusters_REAL.png)

![Elbow and silhouette scan](plot_elbow_silhouette_REAL.png)

## Real segment profile

| Cluster | Recency | Frequency | Monetary | Count | Role |
|---|---|---|---|---|---|
| 0 | 23 days | 10.2 orders | £6,604 | 511 | **Champions** — high-frequency, high-spend, still active |
| 3 | 67 days | 3.0 orders | £1,362 | 1,096 | At-Risk — moderate value, went quiet |
| 1 | 61 days | 1.3 orders | £339 | 861 | Loyal/Core (low order count but recent) |
| 2 | 202 days | 1.3 orders | £303 | 894 | New/Low-Value — one-time or lapsed |

**Headline (real data):** the Champions segment (15.2% of customers) is
predicted to generate **60.4%** of the next 3 months' total revenue — the
same concentration pattern as the synthetic demo, and the number that
actually matters for a retention-budget business case.

## Files in this folder
- `run_real_pipeline.py` — full script (downloads not included; re-fetch the
  raw CSV via the UCI link above, or the GitHub mirror used in this run:
  `https://raw.githubusercontent.com/eaintkyawthmu/UCI_Online_Retail_Dataset_Cleaned_Version/master/Cleaned_UCI_Online_Sale_Dataset.csv`)
- `rfm_features_real.csv` / `rfm_clustered_real.csv` — RFM + cluster labels
- `rfm_clv_predictions_real.csv` — per-customer predicted future CLV
- `segment_business_summary_real.csv` — segment-level business rollup
- `plot_elbow_silhouette_REAL.png`, `plot_pca_clusters_REAL.png`,
  `plot_feature_importance_REAL.png`, `plot_actual_vs_predicted_REAL.png`,
  `plot_actual_vs_predicted_REAL_logspace.png`
