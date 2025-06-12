# CODSOFT

# 💼 CODSOFT Internship - Task 3: Customer Churn Prediction

## 🧠 Problem Statement

Predict whether a customer will **churn (leave)** or **stay** with the company using historical customer data. Churn prediction helps companies identify at-risk customers and take proactive retention steps.

---

## 📁 Dataset

- **Source**: [Kaggle - Bank Customer Churn Dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)
- **File**: `Churn_Modelling.csv`
- **Target column**: `Exited` (1 = churned, 0 = retained)

---

## ⚙️ Technologies Used

- Python
- pandas, numpy
- seaborn, matplotlib
- scikit-learn

---

## 🧪 ML Workflow Overview

1. **Data Preprocessing**:
   - Dropped irrelevant columns: `RowNumber`, `CustomerId`, `Surname`
   - Encoded categorical variables:
     - `Gender` → Label Encoding
     - `Geography` → One-hot encoding
   - Feature scaling using `StandardScaler`

2. **Model Building**:
   - Split into train/test sets (80/20)
   - Trained a **Random Forest Classifier** with 100 estimators

3. **Evaluation**:
   - Classification Report (Precision, Recall, F1-Score)
   - Confusion Matrix Visualization

---

## 🧾 Results Snapshot

```plaintext
              precision    recall  f1-score   support

           0       0.87      0.95      0.91      1607
           1       0.72      0.50      0.59      393

    accuracy                           0.85      2000
   macro avg       0.80      0.72      0.75      2000
weighted avg       0.84      0.85      0.84      2000
