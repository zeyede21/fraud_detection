# 🛡️ Interim Report – Fraud Detection Project (Task 1)

**Author:** Ziden  
**Submission Date:** July 20, 2025  
**Repository:** [GitHub Link Here]

---

## 🔍 1. Dataset Overview

We are analyzing two datasets:

- `Fraud_Data.csv`: Contains user purchase transactions with timestamps, IP addresses, and class labels (0 = legit, 1 = fraud).
- `IpAddress_to_Country.csv`: Maps IP address ranges to country names.

---

## 🧼 2. Data Cleaning & Preprocessing

### ✅ Steps Taken:

- Removed duplicates from both datasets.
- Dropped rows with missing key fields: `user_id`, `purchase_value`, `class`.
- Converted IP addresses to 32-bit integers using `ip_to_int`.
- Mapped IP addresses in `fraud_df` to country names by range joining with `ip_df`.

📁 Cleaned datasets were saved as:
- `data/fraud_cleaned.csv`
- `data/ip_cleaned.csv`

---

## 📊 3. Exploratory Data Analysis (EDA)

### Fraud Class Distribution
![Class Distribution](../visuals/class_distribution.png)

- ~95% of transactions are legitimate, indicating a strong **class imbalance**.

### Purchase Value Distribution
![Purchase Value](../visuals/purchase_value_distribution.png)

- Most purchases fall below \$500, with a few high outliers.

### Fraud Rate by Country
![Fraud Rate by Country](../visuals/fraud_rate_by_country.png)

- Some countries exhibit higher fraud rates, e.g., `Nigeria`, `Pakistan`.

### Monthly Fraud Rate
![Monthly Fraud Rate](../visuals/monthly_fraud_rate.png)

- Fraud appears to vary over time, showing spikes on certain months.

### Purchase Value by Class
![Boxplot](../visuals/purchase_value_by_class.png)

- Fraudulent transactions tend to have slightly higher purchase values.

---

## 🧠 4. Feature Engineering

### 🧮 1. `time_since_signup`
- Created a new feature measuring the seconds between signup and purchase.
- Captures user maturity; new users are more likely to be fraudulent.

![Time Since Signup](../visuals/time_since_signup.png)

### 🗺 2. IP to Country Mapping
- Mapped integer-form IPs to countries using lower/upper IP bounds in `ip_df`.
- Enables geographic fraud analysis.

---

## ⚖️ 5. Class Imbalance

### Observations:
- ~9.4% of transactions are fraudulent (class = 1)
- ~90.6% of transactions are legitimate (class = 0)

### Proposed Handling Strategies:
- Use metrics like **Precision, Recall, F1-score, ROC-AUC** instead of Accuracy.
- Try **SMOTE**, **Random Undersampling**, or **Class Weights** for training.
- Use **Stratified Cross-Validation** to retain class balance in folds.

---

## ✅ Summary

We’ve successfully:
- Cleaned and merged the datasets.
- Engineered important features like `time_since_signup` and `country`.
- Conducted basic EDA with visualizations.
- Identified class imbalance and proposed mitigation strategies.

➡️ Ready for **Task 2: Model Training & Evaluation**!


