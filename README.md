# 🌌 Exoplanet Classification using NASA Kepler Dataset

## Overview

This project uses supervised machine learning to classify **Confirmed Exoplanets** and **False Positives** using NASA's Kepler mission dataset.

The project demonstrates a complete machine learning workflow, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and performance evaluation.

---

## Dataset

* **Source:** NASA Kepler Exoplanet Dataset
* Target Variable: `koi_disposition`
* Classes:

  * Confirmed Planet (1)
  * False Positive (0)

Candidate observations were removed to convert the problem into a binary classification task.

---

## Project Workflow

1. Data Cleaning
2. Missing Value Analysis
3. Median Imputation
4. Feature Selection
5. Train-Test Split
6. Feature Scaling (for Logistic Regression)
7. Model Training
8. Model Evaluation
9. Model Comparison

---

## Models Used

* Logistic Regression
* Random Forest Classifier
* HistGradientBoosting Classifier

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC Curve
* ROC-AUC

---

## Results

| Model                |   Accuracy |
| -------------------- | ---------: |
| Logistic Regression  |        93% |
| Random Forest        |     97.16% |
| HistGradientBoosting | **97.60%** |

HistGradientBoosting achieved the highest overall performance and was selected as the final model.

---

## Visualizations

The project includes:

* Missing Value Analysis
* Correlation Heatmap
* ROC Curve Comparison
* Feature Importance
* Classification Report

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## Project Structure

```text
Exoplanet-Classification/
│
├── Exoplanet_Classification.ipynb
├── README.md
└── images/
```

---

## Future Improvements

* Hyperparameter tuning
* Cross-validation
* Streamlit dashboard
* Model deployment
* Explainability using SHAP values

---

## Key Learning Outcomes

This project helped me gain practical experience in:

* Data preprocessing
* Handling missing values
* Exploratory Data Analysis (EDA)
* Feature selection
* Comparing multiple machine learning models
* Evaluating classification models using ROC-AUC and F1-score
* Building an end-to-end machine learning pipeline
