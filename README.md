# Breast Cancer Classification using Machine Learning

## a. Problem Statement

Breast cancer is one of the leading causes of cancer-related deaths
worldwide. Early and accurate detection is critical for improving
patient survival rates.

The objective of this project is to build and compare multiple machine
learning classification models to predict whether a tumor is:

-   **Malignant (1)**
-   **Benign (0)**

using diagnostic features extracted from digitized images of breast mass
cell nuclei.

This is a **binary classification problem**.

------------------------------------------------------------------------

## b. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Data Set\
**Source:** [Kaggle (UCI ML Repository version)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)\
**Number of Instances:** 569\
**Number of Features:** 30 numerical features\
**Target Variable:** `diagnosis` (M = Malignant, B = Benign)

------------------------------------------------------------------------

## c. Models Used

1.  Logistic Regression\
2.  Decision Tree\
3.  k-Nearest Neighbors (kNN)\
4.  Naive Bayes\
5.  Random Forest (Ensemble)\
6.  XGBoost (Ensemble)

### Model Comparison Table

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| kNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest (Ensemble) | 0.9737 | 0.9929 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| XGBoost (Ensemble) | 0.9737 | 0.9940 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

------------------------------------------------------------------------

## Observations

| ML Model Name | Observation |
|---------------|------------|
| Logistic Regression | Strong performance due to near-linear separability of the dataset. |
| Decision Tree | Lower generalization performance; may suffer from overfitting. |
| kNN | Good performance with proper scaling; slightly lower recall. |
| Naive Bayes | High AUC but lower overall metrics due to feature independence assumption. |
| Random Forest (Ensemble) | Excellent overall performance with perfect precision and strong MCC. |
| XGBoost (Ensemble) | Matched Random Forest accuracy and achieved highest AUC; robust ensemble method. |

------------------------------------------------------------------------
