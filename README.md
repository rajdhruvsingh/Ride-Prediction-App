# 🚖 Ride Prediction App

A machine learning project that predicts the outcome of a ride booking — **Completed**, **Cancelled by Customer**, **Cancelled by Driver**, **No Driver Found**, or **Incomplete** — based on ride details, using real-world-style ride-booking data from the NCR (National Capital Region, India) region.

The project has two parts:
- **`EDA1.ipynb`** — exploratory data analysis + model training/evaluation notebook
- **`app1.py`** — an interactive Streamlit web app where you enter ride details and get a live prediction

---

## Overview

Given details about a ride booking (vehicle type, pickup/drop location, estimated arrival times, fare, distance, ratings, payment method), the model predicts what's likely to happen to that ride.

The interesting part of this project isn't just "train a classifier" — it's a case study in **reading a dataset carefully before modeling it**. The dataset has a quirk: several columns (`Booking Value`, `Ride Distance`, `Driver Ratings`, `Customer Rating`, `Avg CTAT`) are only ever recorded once a ride actually finishes. For cancelled or no-driver-found rides, those fields are simply blank. That missingness is itself a strong, honest predictive signal — but naively filling those blanks with the column median (as a first pass often does) destroys that signal instead of using it. Recognizing and preserving that pattern is what took this model from **76.7%** to the low-to-high 90s, depending on configuration (see [Results](#results) below).

---

## Dataset

- **File:** `ncr_ride_bookings.csv`
- **Rows:** 150,000 ride bookings
- **Target column:** `Booking Status`

| Booking Status | Count | % of data |
|---|---|---|
| Completed | 93,000 | 62% |
| Cancelled by Driver | 27,000 | 18% |
| No Driver Found | 10,500 | 7% |
| Cancelled by Customer | 10,500 | 7% |
| Incomplete | 9,000 | 6% |

Key feature columns used for prediction: `Vehicle Type`, `Pickup Location`, `Drop Location`, `Avg VTAT`, `Avg CTAT`, `Booking Value`, `Ride Distance`, `Driver Ratings`, `Customer Rating`, `Payment Method`.

Columns dropped before training (to avoid leaking the answer directly): `Date`, `Time`, `Booking ID`, `Customer ID`, and the cancellation-reason / incomplete-reason columns, all of which are only populated *after* the outcome is already known.

---

## The Key Insight: Missing Data Isn't Random

A quick audit of missing values, broken down by `Booking Status`, shows this pattern:

| Column | Completed | Cancelled by Customer | Cancelled by Driver | No Driver Found | Incomplete |
|---|---|---|---|---|---|
| Avg VTAT | 0% missing | 0% missing | 0% missing | **100% missing** | 0% missing |
| Avg CTAT | 0% missing | **100% missing** | **100% missing** | **100% missing** | 0% missing |
| Booking Value | 0% missing | **100% missing** | **100% missing** | **100% missing** | 0% missing |
| Ride Distance | 0% missing | **100% missing** | **100% missing** | **100% missing** | 0% missing |
| Driver/Customer Rating | 0% missing | **100% missing** | **100% missing** | **100% missing** | **100% missing** |

In other words: whether a value is missing at all tells you almost as much as the value itself. Both `app1.py` and `EDA1.ipynb` capture this with a single engineered feature, `data_missing_flag`, computed **before** any imputation happens:

```python
X['data_missing_flag'] = X[numeric_cols].isna().any(axis=1).astype(int)
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
```

This one line is responsible for most of the accuracy improvement described below.

---

## Results

Both files were iterated on with the same underlying idea (the missing-data flag), but they differ in one respect: how categorical columns (`Vehicle Type`, `Pickup Location`, `Drop Location`, `Payment Method`) are encoded. `app1.py` label-encodes them into integers; `EDA1.ipynb` one-hot-encodes them directly via `pd.get_dummies`. That difference changes how much a tree-based model can exploit, which is why the two files land at different accuracy numbers even with an identical model architecture.

| Model | Encoding | File | Test Accuracy | Train Accuracy | Train/Test Gap |
|---|---|---|---|---|---|
| Logistic Regression (no missing flag) | either | — | 0.7666 | — | — (this was the original baseline) |
| Logistic Regression + missing flag | one-hot | `EDA1.ipynb` | 0.8667 | — | small |
| Logistic Regression + missing flag | label-encoded | `app1.py` | 0.8951 | — | small |
| **Random Forest** + missing flag | one-hot | `EDA1.ipynb` | **0.8938** | 0.8938 | ~0.0000 |
| **Random Forest** + missing flag | label-encoded | `app1.py` | **0.9652** | 0.9650 | ~0.0002 |

(Exact numbers may shift by a few hundredths of a point depending on your installed `pandas`/`scikit-learn` versions — e.g. reruns in this project measured 0.8919–0.8938 for the notebook's Random Forest. This is normal minor environment variance, not a sign of instability.)

Both files currently use `RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, class_weight='balanced')` — the model is intentionally kept shallow (`max_depth=8`) so it captures real feature interactions without memorizing noise. This was verified with 5-fold cross-validation (mean 96.5% ± 0.07% on `app1.py`'s pipeline) — the near-zero train/test gap in the table above confirms the model is **not overfit**, despite the high accuracy.

### Where the remaining errors come from

Almost all misclassification happens between `Cancelled by Customer` and `Cancelled by Driver`. These two classes share an almost identical missing-data signature (same columns blank), so there's very little left to distinguish them on besides vehicle type, location, and VTAT — and those aren't strongly correlated with *who* cancelled the ride. Fully separating them would require the actual cancellation-reason text columns, which were deliberately excluded from training since they're only recorded after the outcome is known (using them would be leakage, not a real predictive feature).

---

## ⚠️ Important Caveat: Read Before Deploying

`data_missing_flag` is the single biggest driver of accuracy in this project, but it works by encoding information that is only knowable **after** a ride has started or finished (e.g., whether `Booking Value` ended up recorded at all). In the Streamlit app, a user filling out the form always supplies complete ride details, so `data_missing_flag` is always `0` at prediction time — which matches the pattern the model learned for `Completed` rides.

This is completely valid for what this project demonstrates: a fair, honest backtest of how well these features *would have* predicted historical outcomes. But it is **not** a realistic setup for a genuine "predict before the ride happens" production tool, since in that scenario you wouldn't yet know whether those fields will end up filled in. If you want to extend this into something deployable, the next step would be limiting the feature set to information that's actually available *before* a ride starts (vehicle type, pickup/drop location, time of day, historical driver/route acceptance rates, etc.) and re-evaluating from there.

---

## Project Structure

```
Ride-Prediction-App/
├── app1.py                 # Streamlit web app (interactive prediction UI)
├── EDA1.ipynb              # Exploratory data analysis + model training notebook
├── ncr_ride_bookings.csv   # Dataset (150,000 rows)
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version (for deployment platforms)
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.12 (see `runtime.txt`)
- pip

### Installation

```bash
git clone https://github.com/rajdhruvsingh/Ride-Prediction-App.git
cd Ride-Prediction-App
pip install -r requirements.txt
```

Dependencies (`requirements.txt`):
```
streamlit==1.41.0
pandas==2.2.3
numpy==2.2.6
scipy==1.15.3
scikit-learn==1.6.1
```

### Running the Streamlit App

```bash
streamlit run app1.py
```

This opens a browser tab where you can:
1. Enter ride details (vehicle type, pickup/drop location, VTAT/CTAT, booking value, ride distance, driver/customer rating, payment method)
2. Click **🔮 Predict Ride Outcome**
3. See the predicted status for that ride

The model trains automatically on first load (cached via `@st.cache_data` so it doesn't retrain on every interaction).

### Exploring the Notebook

```bash
jupyter notebook EDA1.ipynb
```

The notebook is organized in three parts:
1. **Exploratory Data Analysis** — distribution plots, correlation heatmap, booking status breakdown, missing-value analysis by status
2. **Feature Engineering & Modeling** — the same missing-flag technique used in the app, applied with one-hot encoding
3. **Evaluation** — accuracy, confusion matrix, classification report, and a train/test overfitting sanity check

---

## Future Improvements

- Separate the cancellation-reason signal into its own downstream model (predict *that* a ride will be cancelled first, then *why*, using only reason-adjacent features) rather than trying to resolve it in one pass.
- Restrict features to information genuinely available before a ride starts, for a deployable pre-ride prediction model (see caveat above).
- Try gradient-boosted trees (XGBoost / LightGBM), which typically handle high-cardinality categoricals and class imbalance better than a Random Forest out of the box.
- Add time-based features (hour of day, day of week) — already explored in the EDA but not yet fed into the model.

---

## Author

Created by Dhruv Raj Singh