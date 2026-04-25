# Loan Default Prediction using Machine Learning and Regularization-Based Model Selection

This project develops a credit-risk classification pipeline to predict loan default using borrower, loan, credit-history, and property-level characteristics.

The project emphasizes not only predictive performance, but also proper model validation, class imbalance handling, leakage diagnostics, and regularization-based model selection.

## Objective

To predict whether a borrower defaults on a loan and compare different classification models under a consistent validation framework.

The main goals are:

- Predict loan default status
- Identify relevant financial and credit-risk predictors
- Compare linear, regularized, and tree-based classifiers
- Handle class imbalance using weighted models
- Diagnose and correct data leakage
- Evaluate models using out-of-sample performance metrics

## Dataset

The dataset contains loan-level information on borrowers and applications, including:

- Borrower demographics
- Income and employment information
- Loan amount and loan-to-value ratios
- Interest rate and upfront charges
- Credit history and risk indicators
- Property characteristics
- Default status

The target variable is:

- `Status = 0`: no default
- `Status = 1`: default

## Methodology

The notebook follows a full supervised-learning pipeline:

1. Data loading and cleaning
2. Missing-value diagnostics
3. Exploratory data analysis
4. Structure-aware imputation
5. Outlier treatment
6. Categorical encoding
7. Feature engineering
8. Train/validation/test split
9. Class imbalance handling
10. Regularized model selection with cross-validation
11. Model comparison
12. Leakage detection and correction
13. Final out-of-sample evaluation

## Models

The project compares:

- Logistic Regression
- LASSO Logistic Regression
- Ridge Classifier
- Decision Tree
- Random Forest
- XGBoost

## Evaluation Metrics

Given the class imbalance in loan default prediction, the project evaluates models using:

- ROC-AUC
- F1-score
- Precision
- Recall
- Accuracy
- Confusion matrices

## Key Technical Features

- Regularization-based model selection
- Stratified cross-validation
- Class-weighted classification
- Leakage diagnostics
- Train/validation/test separation
- Credit-risk feature engineering
- Interpretable comparison between linear and nonlinear models

## Tools

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- matplotlib
- seaborn

## Repository Structure

```text
loan-default-prediction-regularization-model-selection/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── loan_default_prediction_regularization.ipynb
└── data/
    └── Loan.csv
