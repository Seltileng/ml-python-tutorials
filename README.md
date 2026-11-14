# AIMLCLASS: Applied Machine Learning & Data Science Coursework

A comprehensive collection of machine learning workflows, exploratory data analysis (EDA) pipelines, regression models, and classification algorithms implemented in Python using Jupyter Notebooks and `scikit-learn`.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dataset Catalog](#dataset-catalog)
- [Notebooks & Project Details](#notebooks--project-details)
  - [1. Diabetes EDA & Preprocessing (`diabetes.ipynb`)](#1-diabetes-eda--preprocessing-diabetesipynb)
  - [2. House Price Prediction via Linear Regression (`house_price_regression.ipynb`)](#2-house-price-prediction-via-linear-regression-house_price_regressionipynb)
  - [3. Baseball Hall of Fame Classification (`sporting_hits.ipynb`)](#3-baseball-hall-of-fame-classification-sporting_hitsipynb)
  - [4. Student Pass/Fail Probability (`student_pass-fail_proba_LogisticRegression.ipynb`)](#4-student-passfail-probability-student_pass-fail_proba_logisticregressionipynb)
  - [5. Standalone Scripts (`import pandas as pd.py`)](#5-standalone-scripts-import-pandas-as-pdpy)
- [Machine Learning Concepts & Methodologies](#machine-learning-concepts--methodologies)
- [Setup & Installation](#setup--installation)
- [Tech Stack](#tech-stack)

---

## Overview

This repository serves as a hands-on laboratory for core machine learning concepts, transitioning from foundational exploratory data analysis and feature engineering to predictive modeling and probabilistic inference.

Key focus areas include:
- **Exploratory Data Analysis (EDA)**: Five-number summaries, missing-value diagnostics, multi-feature boxplot distributions, correlation matrices, and class balance visualizations.
- **Data Preprocessing & Scaling**: Outlier detection via the Interquartile Range (IQR) rule, Min-Max Normalization ($\in [0, 1]$), and Standard Z-score Standardization ($\mu=0, \sigma=1$).
- **Regression Modeling**: Multiple Linear Regression predicting continuous real estate prices with $R^2$ score and Mean Absolute Error (MAE) evaluation alongside diagonal actual-vs-predicted residual plotting.
- **Binary Classification**: Logistic Regression applied to sports career milestones and academic outcomes, evaluated via Accuracy, Confusion Matrix, Precision, Recall, $F_1$-score, Log-odds / Odds Ratios, and continuous Sigmoid probability curves.

---

## Repository Structure

```text
AIMLCLASS/
├── diabetes.ipynb                                 # EDA, IQR outlier filtering, correlation heatmap, feature scaling
├── house_price_regression.ipynb                  # House price prediction (Linear Regression, R2, MAE, evaluation plots)
├── house_pricing.ipynb                           # House price regression variant (80/20 train/test split, residual analysis)
├── house_pricing_prediction_LinearRegression.ipynb# House price linear regression model pipeline
├── sporting_hits.ipynb                           # Baseball Hall of Fame binary classification & Odds Ratio extraction
├── student_pass-fail_proba_LogisticRegression.ipynb # Student pass/fail logistic regression, sigmoid curve & probability inference
├── import pandas as pd.py                        # Python utility script demonstrating dataset ingestion
│
├── 500hits.csv                                   # Baseball player career hitting statistics and Hall of Fame status
├── diabetes.csv.xls                              # Pima Indians Diabetes diagnostic dataset
├── House_Pricing_Regression_300_Rows.csv          # Real estate dataset with architectural features & property prices
├── House_Pricing_Regression_300_Rows-1.csv        # Alternative version of the 300-row house pricing dataset
├── student_pass_fail_classification.csv          # Student study hours vs. exam pass/fail outcome
├── student_scores (3).csv                        # Student study hours vs. raw test scores
│
├── requirements.txt                              # Python library dependencies
└── README.md                                     # Comprehensive project documentation
```

---

## Dataset Catalog

| Dataset File | Domain | Problem Type | Target Variable | Key Features | Records |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`diabetes.csv.xls`** | Healthcare | EDA / Preprocessing | `Outcome` (0: Non-Diabetic, 1: Diabetic) | `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `Age` | 768 rows, 9 cols |
| **`House_Pricing_Regression_300_Rows.csv`** | Real Estate | Regression | `Price_NGN` (Continuous in NGN) | `Bedrooms`, `Bathrooms`, `Square_Footage`, `Year_Built`, `Location` | 300 rows, 7 cols |
| **`House_Pricing_Regression_300_Rows-1.csv`** | Real Estate | Regression | `Price_NGN` (Continuous in NGN) | `Bedrooms`, `Bathrooms`, `Square_Footage`, `Year_Built`, `Location` | 300 rows, 7 cols |
| **`500hits.csv`** | Sports Analytics | Binary Classification | `HOF` (1: Hall of Fame, 0: Not Inducted) | `YRS`, `G`, `AB`, `R`, `H`, `2B`, `3B`, `HR`, `RBI`, `BB`, `SO`, `SB`, `BA` | 465 rows, 16 cols |
| **`student_pass_fail_classification.csv`** | Education | Binary Classification | `Pass_Fail` (1: Pass, 0: Fail) | `Study_Hours` | 30 rows, 2 cols |
| **`student_scores (3).csv`** | Education | Ingestion / Regression | `Scores` (Numerical Exam Score) | `Hours` | 25 rows, 2 cols |

---

## Notebooks & Project Details

### 1. Diabetes EDA & Preprocessing (`diabetes.ipynb`)

Focuses on diagnostic data analysis, cleaning, and transformation on the Pima Indians Diabetes dataset:
- **Structural Exploration**: Inspecting non-null data types (`df.info()`), descriptive statistics (`df.describe()`), and null values (`df.isnull().sum()`).
- **Outlier Diagnostics via Boxplots**: Grid of horizontal boxplots across all 9 features identifying significant outlier spread, especially in `Insulin` and `SkinThickness`.
- **IQR Outlier Removal**:
  $$\text{IQR} = Q_3 - Q_1$$
  $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
  Applied to filter extreme insulin measurements into a sanitized subset `clean_df`.
- **Correlation Analysis**: Pearson correlation matrix rendered using `seaborn.heatmap`, revealing `Glucose` and `BMI` as top correlates with diabetic `Outcome`.
- **Target Distribution**: Cleaned pie chart visualization showing class balance (65% non-diabetic vs. 35% diabetic).
- **Feature Scaling**: Side-by-side comparison of:
  - **Min-Max Normalization**: Scales features to $[0, 1]$:
    $$X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$
  - **Standardization (Z-Score)**: Transforms features to zero mean and unit variance:
    $$X_{\text{std}} = \frac{X - \mu}{\sigma}$$

---

### 2. House Price Prediction via Linear Regression (`house_price_regression.ipynb`)

Implements an end-to-end Ordinary Least Squares (OLS) Linear Regression pipeline predicting house prices in Nigerian Naira (`Price_NGN`):
- **Feature Matrix Preparation**: Drops non-predictive `House_ID` and separates the target variable `Price_NGN`.
- **Train/Test Splitting**: 80% training set ($N=240$) and 20% testing set ($N=60$) with `random_state=42`.
- **Model Fitting**: Trains `sklearn.linear_model.LinearRegression` on multiple physical property attributes.
- **Model Evaluation**:
  - **$R^2$ Score (Coefficient of Determination)**: Quantifies the proportion of price variance explained by the features:
    $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  - **Mean Absolute Error (MAE)**: Measures average absolute monetary deviation:
    $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
- **Residual Visualization**: Scatter plot contrasting actual test prices against predicted prices along a $45^\circ$ ideal fit diagonal ($y = x$).

*Companion notebooks `house_pricing.ipynb` and `house_pricing_prediction_LinearRegression.ipynb` provide alternative implementations and validation runs.*

---

### 3. Baseball Hall of Fame Classification (`sporting_hits.ipynb`)

Applies binary Logistic Regression to predict whether elite baseball hitters qualify for the National Baseball Hall of Fame (`HOF`):
- **Career Feature Selection**: Uses 14 cumulative metrics including At-Bats (`AB`), Hits (`H`), Home Runs (`HR`), Batting Average (`BA`), Runs Batted In (`RBI`), and Strikeouts (`SO`).
- **Model Training**: Configured with `max_iter=1000` to guarantee numerical convergence on multi-scale features.
- **Comprehensive Evaluation**:
  - **Accuracy**: Overall classification rate.
  - **Confusion Matrix**: Quantitative breakdown of True Negatives, False Positives, False Negatives, and True Positives.
  - **Precision & Recall**: Evaluates false alarm penalty vs. detection capability for Hall of Fame induction.
  - **$F_1$-Score**: Harmonic mean providing balanced evaluation on class-imbalanced data.
- **Statistical Interpretation via Odds Ratios**:
  - Extracts model weights $\beta_i$ and intercept $\beta_0$.
  - Exponentiates weights to compute Odds Ratios:
    $$\text{Odds Ratio} = e^{\beta_i}$$
  - An Odds Ratio $> 1$ indicates higher likelihood of Hall of Fame entry per additional unit of the statistic.

---

### 4. Student Pass/Fail Probability (`student_pass-fail_proba_LogisticRegression.ipynb`)

Demonstrates probabilistic modeling of student examination success based on study duration:
- **Stratified Partitioning**: Employs `stratify=y` to ensure balanced pass/fail ratios across training and test subsets.
- **Binary Logistic Regression**: Models the probability of passing ($Y=1$) as a function of study hours ($X$):
  $$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}$$
- **Probabilistic Inference (`predict_proba`)**:
  - Calculates confidence intervals for test set predictions.
  - Predicts pass probability for custom inputs (e.g., student studying 7 hours).
- **Dual Visualizations**:
  - Actual vs. predicted discrete classification scatter plot.
  - Smooth continuous Sigmoid curve across all study hour ranges with an annotated $0.5$ decision boundary.

---

### 5. Standalone Scripts (`import pandas as pd.py`)

A standalone script demonstrating tabular data ingestion from `student_scores (3).csv` using `pd.read_csv`, verifying column schemas (`Hours`, `Scores`) and initial records.

---

## Machine Learning Concepts & Methodologies

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Machine Learning Pipelines                      │
├───────────────────────────────────┬────────────────────────────────────┤
│         Supervised Regression     │      Supervised Classification     │
│   (House Price Prediction)        │  (Diabetes, Hall of Fame, Pass/Fail│
├───────────────────────────────────┼────────────────────────────────────┤
│ • Objective: Continuous variable  │ • Objective: Discrete category     │
│ • Model: Linear Regression (OLS)  │ • Model: Logistic Regression       │
│ • Evaluation: R², MAE             │ • Evaluation: Accuracy, Precision, │
│ • Target: Price (NGN)             │   Recall, F1, Confusion Matrix     │
│ • Plotting: Scatter vs y = x line │ • Plotting: Sigmoid curve, Pie, CM │
└───────────────────────────────────┴────────────────────────────────────┘
```

### Preprocessing Checklist
1. **Missing Data Auditing**: Checking dataset integrity using `isnull().sum()`.
2. **Outlier Mitigation**: Utilizing Tukey's 1.5 IQR fences to prevent leverage distortion.
3. **Feature Scaling**: Selecting between Min-Max scaling (when bounds $[0, 1]$ are required) and Z-score standardization (when zero-centered normal distribution is assumed).
4. **Stratified Splitting**: Preserving class proportions during train/test splits for classification tasks.

---

## Setup & Installation

### 1. Prerequisites
- Python 3.8+
- Recommended: `pip` and Python virtual environment (`venv`)

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/AIMLCLASS.git
cd AIMLCLASS
```

### 3. Create and Activate Virtual Environment
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Launch Notebooks
```bash
jupyter notebook
# or
jupyter lab
```

---

## Tech Stack

- **Language**: Python 3.8+
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn` (`LinearRegression`, `LogisticRegression`, `MinMaxScaler`, `StandardScaler`, `train_test_split`, `metrics`)
- **Data Visualization**: `matplotlib`, `seaborn`
- **Interactive Computing**: `jupyter` / `IPython`
