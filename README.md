# ⚡ ChurnSight — Customer Retention Intelligence

> **Live demo →** [churnsight](https://churn-telco--ds-nvs.streamlit.app/)

A full end-to-end machine learning project — from raw data to a deployed retention dashboard — predicting customer churn for a telecom company using XGBoost, K-Means segmentation, and survival analysis.

---

## 🎯 Business Problem

Telecom companies lose 15–25% of customers annually. Acquiring a new customer costs 5–7× more than retaining an existing one. This project builds a system that:

1. **Predicts** which customers are likely to churn (and with what probability)
2. **Segments** customers by risk profile and lifetime value
3. **Prioritizes** who is worth a retention campaign based on expected ROI

---

## 📊 Results

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.8402** [0.8138 – 0.8667] 95% CI |
| PR-AUC | 0.6172 |
| F-β=2.0 | 0.7441 @ threshold 0.151 |
| Brier Score | 0.1402 |
| Customers scored | 7,043 |
| CV–Test gap | +0.0066 (healthy generalization) |

---

## 🏗️ Project Architecture

```
Raw Data (IBM Telco)
       ↓
Phase 1 — EDA & Cleaning
  • Null handling, type enforcement
  • Feature engineering (Num_Services, BillRatio, ChargeVelocity)
  • Distribution analysis & churn drivers visualization
       ↓
Phase 2 — Churn Prediction Model
  • Benchmarked: Logistic Regression, XGBoost, LightGBM, TabNet
  • Winner: XGBoost (calibrated with CalibratedClassifierCV)
  • Threshold optimization via F-β=2.0 (recall-weighted for business)
  • SHAP explainability on test set
       ↓
Phase 3 — Segmentation & LTV
  • K-Means clustering (k=4, silhouette-optimized)
  • Kaplan-Meier survival curves by segment
  • Cox Proportional Hazards for time-to-churn
  • ROI-based intervention scoring (EV = prob × LTV − campaign_cost)
       ↓
Streamlit Dashboard (this app)
  • Real-time scoring for individual customers
  • Segment assignment + radar profile
  • Business case: EV waterfall, revenue at risk
```

---

## 🔑 Top Predictive Features (SHAP)

1. `Has_Contract` — Month-to-month customers churn at 3× the rate of contracted ones
2. `BillRatio` — High ratio signals new customers before commitment solidifies
3. `InternetService_Fiber optic` — Fiber customers have higher charges and expectations
4. `tenure` — First 12 months are the highest-risk window
5. `MonthlyCharges` — Non-linear relationship with churn at the high end

---

## 🧩 Customer Segments

| Segment | Profile | Action |
|---------|---------|--------|
| 🚨 High-Value At-Risk | High spend, high churn probability | Immediate premium retention offer |
| ⚠️ Low-Value At-Risk | Low spend, high churn probability | Evaluate ROI before intervening |
| ⚓ Loyal Anchors | Long tenure, contracted | Upsell & deepen adoption |
| 🌱 New & Vulnerable | Early lifecycle, uncommitted | 90-day onboarding nurture |

---

## 🚀 Run Locally

```bash
git clone https://github.com/TU_USUARIO/tu-repo.git
cd tu-repo
pip install -r requirements.txt
streamlit run app.py
```

---

## 🗂️ Repository Structure

```
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_baseline_model_churn.ipynb
│   └── 03_segmentation_churn_LTV.ipynb
├── models/
│   ├── phase2_churn_model.pkl    # Calibrated XGBoost
│   ├── phase2_scaler.pkl
│   ├── phase2_metadata.json      # Features, threshold, business params
│   ├── phase3_kmeans.pkl         # K-Means (k=4)
│   ├── phase3_cluster_scaler.pkl
│   └── phase3_metadata.json      # Segment names, churn rates
├── app.py                        # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

`Python` `XGBoost` `LightGBM` `scikit-learn` `pandas` `SHAP` `Kaplan-Meier` `Cox PH` `Streamlit` `Plotly` `K-Means`

---

*Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) · 7,043 customers · 21 features*

---

*made with ♥ by Nevets Agetro*