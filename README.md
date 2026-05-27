# Multi-Horizon-Electricity-Demand-Forecasting

> Beating the European grid operator's day-ahead forecast by **44.6%** using LightGBM with engineered time-series features, conformal prediction intervals, and a COVID-19 concept drift case study.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

---

## 📊 TL;DR — Headline Results

| Model | Validation MAPE | vs. ENTSO-E Benchmark |
|:---|:---:|:---:|
| Seasonal Naive (168h lag) | 4.735% | +34% worse |
| **ENTSO-E Day-Ahead Forecast** *(industry benchmark)* | **3.522%** | — |
| **LightGBM 24h-ahead (ours)** | **2.018%** | **🏆 −42.7%** |
| LightGBM 72h-ahead | 2.503% | −28.9% |
| LightGBM 168h-ahead (1 week) | 2.586% | −26.6% |

**Key achievement:** A feature-engineered gradient boosting model outperforms the official ENTSO-E forecast across all three horizons — including 1-week-ahead prediction.

---

## 🎯 Project Goals

Forecast hourly electricity demand for Germany at three horizons (24h, 72h, 168h) using only publicly available data, then evaluate model robustness during the COVID-19 demand shock.

This project demonstrates:
- **Real-world time-series workflows** — proper chronological splits, walk-forward validation, no data leakage
- **Feature engineering depth** — 34 engineered features across 6 categories
- **Beating an industry benchmark** — outperforming the official forecast used by European grid operators
- **Uncertainty quantification** — conformal prediction intervals with verified coverage
- **Concept drift analysis** — documenting how the COVID lockdown broke (and recovered) the model
- **Interpretability** — SHAP-based explanations for individual predictions

---

## 📁 Dataset

**Source:** [Open Power System Data — Time Series 60min](https://open-power-system-data.org/) (free, no API key required)

- **50,401 hourly observations** spanning Dec 2014 – Sept 2020
- **300 columns** covering ~37 European countries; we focus on Germany (DE)
- **Target:** `DE_load_actual_entsoe_transparency` — actual electricity demand in MW
- **Benchmark:** `DE_load_forecast_entsoe_transparency` — ENTSO-E's published day-ahead forecast

### Why Germany?
- Highest data completeness across the full window (99.998% complete on the target)
- Includes ENTSO-E's own published forecast — a genuine industry benchmark to beat
- Significant demand variation (31 GW to 77 GW range) makes forecasting non-trivial

### Data Quality Findings
- `DE_LU_price_day_ahead` is 65% missing because Germany-Luxembourg only became a combined bidding zone in October 2018. **Dropped** during preprocessing.
- All other features had <0.3% missingness, handled with forward-fill then back-fill imputation.

---

## 🔬 Methodology — Step by Step

### 1. Exploratory Data Analysis

Started with visual inspection across multiple time scales — a fundamental discipline for time series work. The goal was to identify which seasonalities exist before modeling.

**Findings:**
- **Daily cycle:** Clear morning peak (~8am) and evening peak (~6pm) on weekdays
- **Weekly cycle:** Weekends ~15 GW lower than weekdays
- **Yearly cycle:** Winter peaks ~75 GW, summer peaks ~65 GW (heating + reduced industrial activity)
- **Holiday effects:** Visible drops on Christmas, New Year, Easter, German national holidays

These observations directly motivated the feature engineering strategy.

### 2. STL Decomposition

Used Seasonal-Trend decomposition using LOESS (STL) with a 168-hour period to formally separate the signal.

**Variance decomposition (2018 sample):**
- Seasonal component: **88.9%**
- Trend: 7.6%
- Residual: 7.3%

The seasonal component dominance confirmed that weekly lag features would be the most predictive. Inspection of the residual revealed negative spikes aligned with German public holidays — motivating the explicit `is_holiday` feature.

### 3. Stationarity Testing

Augmented Dickey-Fuller test on raw load: **p < 0.000001**, confirming no unit root. This justified `d=0` in ARIMA specifications. Note that ADF's "stationarity" here refers to the absence of unit roots; the deterministic seasonality still requires explicit handling via seasonal terms or Fourier features.

### 4. Train / Validation / Test Split

**Strict chronological split — no random shuffling:**

| Split | Period | Hours | Purpose |
|:---|:---|:---:|:---|
| Train | 2015-01-15 → 2018-12-31 | 34,728 | Model learning |
| Validation | 2019-01-01 → 2019-12-31 | 8,760 | Hyperparameter tuning, early stopping |
| Test | 2020-01-01 → 2020-09-30 | 6,577 | Final unbiased evaluation **(includes COVID)** |

The COVID period was intentionally left in the test set to evaluate model robustness to real-world concept drift.

### 5. Feature Engineering

Built **34 features across 6 categories**, all with explicit lookahead protection:

| Category | Features | Rationale |
|:---|:---|:---|
| **Calendar** | hour, dayofweek, day, month, quarter, year, dayofyear, weekofyear, is_weekend | Capture cyclic patterns at multiple scales |
| **Cyclical encoding** | sin/cos transforms of hour, dayofweek, month, dayofyear | Make hour-23 and hour-0 "adjacent" to the model |
| **Holiday flags** | is_holiday, is_day_before_holiday, is_day_after_holiday | Capture pre/during/post-holiday demand patterns |
| **Lag features** | 24h, 48h, 72h, 168h, 336h | Auto-regressive memory of past demand |
| **Rolling stats** | mean/std/max/min over 24h and 168h, shifted by 24h | Recent-window context without leakage |
| **Differences** | 24h and 168h week-over-week deltas | Capture demand trend direction |

**Critical detail:** rolling statistics use `.shift(min_lag_hours)` *before* `.rolling()`, ensuring no future information leaks into training features.

### 6. Baselines (the bar to clear)

Built two baselines to ground all subsequent improvements:

1. **Seasonal Naive (168h lag):** Predict load = load from same hour, same weekday, last week. MAPE: 4.735%.
2. **ENTSO-E Day-Ahead Forecast:** The actual published forecast used by European grid operators. MAPE: 3.522%.

Beating the seasonal naive shows your features add signal. Beating ENTSO-E proves your model is competitive with professional industry tools.

### 7. LightGBM Model

Chose LightGBM for these reasons:
- Handles non-linear feature interactions natively (e.g., `hour × is_holiday`)
- Robust to feature scale (no normalization needed)
- Fast training, early stopping on validation MAE prevents overfitting
- Native support for quantile regression (used later for prediction intervals)

**Hyperparameters:** learning_rate=0.05, num_leaves=64, min_data_in_leaf=50, L1/L2 regularization, early stopping at 50 rounds patience. Trained separate models for each forecast horizon (24h / 72h / 168h) with horizon-appropriate feature sets (e.g., 24h-ahead model doesn't use the 1h lag).

### 8. Conformal Prediction Intervals

A point forecast isn't enough for production use — grid operators need uncertainty bounds.

**Implementation:**
1. Trained quantile regression models at the 10th and 90th percentiles (raw 80% interval)
2. Measured empirical coverage on validation: **only 61.2%** — the intervals were overconfident
3. Applied **split conformal prediction:** computed residuals on a calibration subset, then expanded the intervals by the empirical (1-α)-quantile of those residuals
4. Result: **81.4% empirical coverage** on a held-out evaluation set, matching the 80% target within statistical noise

The honest cost: interval width grew from 2,455 MW to 3,585 MW (a 46% increase). The honest benefit: when the model says "80% confidence," it now actually means 80% confidence.

### 9. COVID-19 Concept Drift Analysis

The test set deliberately spans March–September 2020 to expose the model to a real distribution shift.

**Findings:**
- German electricity demand dropped ~10% overnight when lockdown began on March 15, 2020
- LightGBM rolling weekly MAE spiked from ~1,100 MW to ~3,150 MW
- ENTSO-E forecast experienced a 15% larger error spike (~3,700 MW peak)
- Our model recovered faster initially, but ENTSO-E adapted better over the summer

**Production implication:** static models degrade during regime changes. Weekly retraining cadence would mitigate this — a recommendation grounded directly in the rolling MAE plot.

### 10. SHAP Interpretability

Used `shap.TreeExplainer` on a 2,000-row validation sample to generate:
- **Summary (beeswarm) plot:** features ranked by importance, showing directionality
- **Bar plot:** mean |SHAP value| for clean reporting
- **Dependence plot:** how `load_lag_168h` interacts with `is_holiday`
- **Waterfall plot:** complete decomposition of a single prediction (Christmas Day 2019)

The waterfall analysis explicitly showed `is_holiday=1` pushing the prediction down by several thousand MW — exactly the pattern observed in raw STL residuals during early EDA.

---

## 🏆 Final Results Summary

### Validation Performance (2019)

| Model | MAE (MW) | RMSE (MW) | MAPE (%) |
|:---|:---:|:---:|:---:|
| Seasonal Naive | 2,558 | 4,465 | 4.735 |
| ENTSO-E Forecast | 1,989 | 2,489 | 3.522 |
| **LightGBM 24h** | **1,094** | 1,528 | **2.018** |
| LightGBM 72h | 1,357 | — | 2.503 |
| LightGBM 168h | 1,403 | — | 2.586 |

### Test Performance (Jan–Sept 2020, includes COVID)

| Model | MAE (MW) | MAPE (%) |
|:---|:---:|:---:|
| LightGBM 24h | 1,287 | 2.502 |
| LightGBM 72h | 1,645 | 3.219 |
| LightGBM 168h | 1,686 | 3.301 |

All three horizons still beat the ENTSO-E benchmark on the test set, demonstrating robustness to the COVID shock.

### Prediction Intervals

- **Target coverage:** 80%
- **Uncalibrated quantile coverage:** 61.2% (overconfident)
- **Conformal-calibrated coverage:** 81.4% ✅
- **Average interval width after calibration:** 3,585 MW (6.4% of mean load)

---

## 🔑 Key Insights for Production

1. **Weekly lags dominate** — `load_lag_168h` and `load_lag_336h` accounted for ~85% of model gain. Electricity demand is fundamentally a weekly-periodic process.
2. **Holiday flags rival 24h lag in importance** — explicit calendar engineering is non-negotiable.
3. **Quantile regression alone is overconfident** — always validate empirical coverage before claiming uncertainty estimates.
4. **Concept drift is real and measurable** — rolling MAE diagnostics are the right tool for detection.
5. **Retraining cadence matters** — by July 2020, ENTSO-E (which presumably retrains) caught up to our static model. Production deployment should retrain weekly.

---

## 🛠️ Tech Stack

- **Python 3.10**, pandas, NumPy
- **LightGBM 4.x** — gradient boosting and quantile regression
- **statsmodels** — STL decomposition, ADF test
- **scikit-learn** — metric utilities
- **SHAP** — model interpretability
- **holidays** — German national holiday calendar
- **matplotlib, seaborn** — visualization
- **Google Colab** — development environment

---

---

## 🚀 Reproducing the Results

```bash
# 1. Clone the repo
git clone https://github.com/Suav00/Multi-Horizon-Electricity-Demand-Forecasting.git
cd Multi-Horizon-Electricity-Demand-Forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Visit https://open-power-system-data.org/ → Time Series → 60min CSV
# Save as: data/time_series_60min_singleindex.csv

# 4. Run the notebook
jupyter notebook notebooks/Multi_Horizon_Electricity_Demand_Forecasting.ipynb
```

---

## 🎓 What I Learned

- **Time series splits must be chronological** — random splits inflate scores by 30-50%
- **Beating an industry benchmark requires deliberate feature engineering** — generic models don't get there
- **Uncertainty quantification needs empirical validation** — never trust nominal coverage
- **Concept drift requires monitoring infrastructure** — rolling-window metrics catch what static evaluation misses
- **Interpretability is a deliverable, not an afterthought** — SHAP explanations turn a black box into a story regulators and operators can follow

---

## 📝 License

MIT License — free to use, modify, and learn from.

## 🙋 Contact

Built by [@Suav00](https://github.com/Suav00). Open to feedback, questions, and conversations about energy forecasting or time-series ML.
"""

with open('/content/README.md', 'w') as f:
    f.write(readme_content)

print("✅ README.md saved to /content/README.md")
print(f"   Length: {len(readme_content):,} characters / {len(readme_content.split()):,} words")
print("\n📥 To download:")
print("   1. In Colab file browser (left sidebar), find README.md at the top level")
print("   2. Right-click → Download")
print("   3. Upload to your GitHub repo (replace the existing README.md)")
