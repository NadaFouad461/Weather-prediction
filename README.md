#  Rain in Australia - Weather Prediction Model

An end-to-end Machine Learning classification project designed to predict whether it will rain tomorrow in Australia based on daily weather measurements.

---

## 📌 Project Overview
This project focuses on building a binary classification pipeline to address a common meteorological forecasting challenge: **Rain Tomorrow Prediction**. 

Using historical weather data from various Australian weather stations, the project demonstrates a complete data science workflow, including data preprocessing, feature engineering, handling class imbalance, and evaluating machine learning models using business-relevant classification metrics.

---

## 📊 Dataset Description
The dataset contains approximately 10 years of daily weather observations from many locations across Australia.

* **Source:** Weather Australia Dataset
* **Target Variable:** `RainTomorrow` (Binary: `Yes` / `No`)
* **Key Features:**
  * `MinTemp`, `MaxTemp`: Temperature variations
  * `Rainfall`: Amount of rain recorded for the day (mm)
  * `Humidity9am`, `Humidity3pm`: Relative humidity levels
  * `Pressure9am`, `Pressure3pm`: Atmospheric pressure (hPa)
  * `WindGustSpeed`, `WindDir`: Wind metrics and directions
  * `RainToday`: Whether it rained today (`Yes` / `No`)

---

##  Tech Stack & Libraries
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn
  * `LogisticRegression`
  * `StandardScaler`
  * `train_test_split`
* **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

---

## 🔄 Machine Learning Workflow

1. **Exploratory Data Analysis (EDA):**
   * Investigated feature distributions, missing value percentages, and correlations with `RainTomorrow`.
   * Analyzed the class imbalance in the target variable (`No` rain is far more frequent than `Yes`).

2. **Data Preprocessing & Cleaning:**
   * Handled missing values using median imputation for numerical attributes and mode imputation for categorical attributes.
   * Scaled numerical features using `StandardScaler` to optimize gradient descent performance for Logistic Regression.
   * Encoded categorical features (`RainToday`, `WindDir`, etc.) using One-Hot Encoding.

3. **Model Development & Evaluation:**
   * Built a baseline **Logistic Regression** model using Scikit-Learn `Pipeline` to avoid data leakage.
   * Handled class imbalance using `class_weight='balanced'` to improve sensitivity towards rainy days.
   * Evaluated model performance beyond basic accuracy, focusing heavily on **Recall** and **F1-Score** to minimize false negatives (failing to predict rain).

---

> **Key Insight:** Adjusting classification thresholds and incorporating class weights significantly improved the model's **Recall**, ensuring better detection of actual rainy days compared to an unweighted baseline model.

---
