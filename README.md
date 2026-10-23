# Porto Seguro – Insurance Claim Prediction

### End-to-End Machine Learning | Binary Classification | Ensemble Learning

This project investigates the prediction of automobile insurance claim probabilities using the **Porto Seguro Safe Driver Prediction** dataset.

The objective is to develop, optimize, and compare machine learning models capable of identifying policyholders who are more likely to file an insurance claim, while addressing challenges such as class imbalance, anonymized features, and model generalization.

The project was developed as part of the **Advanced Data Analytics** course at Justus Liebig University Giessen.

---

## 1. Project Objectives

- Explore and understand the structure of an anonymized insurance dataset.
- Investigate missing values, feature distributions, and class imbalance.
- Implement appropriate preprocessing and feature engineering techniques.
- Train and optimize different classification models.
- Evaluate predictive performance using suitable metrics.
- Investigate whether ensemble learning improves individual model performance.

## 2. Dataset

**Dataset:** Porto Seguro Safe Driver Prediction

**Source:** Kaggle / OpenML (Data ID: 42742)

The dataset contains anonymized information about automobile insurance policyholders.

The target variable is binary:

- `0`: No insurance claim
- `1`: Insurance claim

### Main Challenges

- **Class imbalance:** Claim cases represent a small proportion of observations.
- **Anonymized features:** Most variables do not have directly interpretable business meanings.
- **Missing values:** Some features contain missing information encoded using special values.
- **Mixed feature types:** The dataset contains numerical, binary, categorical, and calculated variables.

These characteristics make the dataset suitable for investigating classification algorithms and robust model evaluation.

---

## 3. Methodology

The project follows a structured machine learning workflow:

1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Model Training
4. Hyperparameter Optimization
5. Ensemble Learning
6. Model Evaluation and Comparison

### 3.1 Exploratory Data Analysis

The exploratory analysis focuses on:

- Target class distribution
- Missing value patterns
- Numerical and categorical feature distributions
- Correlations between variables
- Identification of potentially irrelevant features

### 3.2 Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Encoding categorical features
- Feature scaling where required
- Stratified train-validation splitting
- Investigation of feature groups such as `ps_calc`

Preprocessing decisions were adapted to the requirements of the individual models.

---

## 4. Machine Learning Models

Three main classification approaches were investigated.

### 4.1 Logistic Regression

Logistic Regression serves as an interpretable linear baseline.

The experiments investigate:

- Regularization
- Class weighting
- Feature preprocessing
- Hyperparameter optimization

### 4.2 XGBoost

XGBoost is used to investigate nonlinear relationships and feature interactions through gradient-boosted decision trees.

The experiments include:

- Tree complexity control
- Learning rate optimization
- Regularization
- Hyperparameter search
- Model performance evaluation

### 4.3 Linear Support Vector Machine

A Linear SVM provides an additional linear classification approach.

The experiments focus on:

- Feature scaling
- Regularization strength
- Class imbalance
- Comparison with Logistic Regression

---

## 5. Ensemble Learning

Beyond individual models, three ensemble strategies were investigated:

**Simple Averaging**

Combining model predictions using equal weights.

**Weighted Averaging**

Combining model predictions using different weights to investigate whether stronger models should contribute more to the final prediction.

**Stacking**

Training a meta-model using predictions from base models.

The objective is to determine whether combining different classifiers improves predictive performance.

---

## 6. Model Evaluation

The main evaluation metrics include:

- **ROC-AUC:** Measures ranking performance across classification thresholds.
- **Normalized Gini Coefficient:** Derived from ROC-AUC as `2 × AUC - 1`.
- **Average Precision:** Summarizes precision-recall performance, particularly relevant for imbalanced classification.

### Model Comparison

| Model | ROC-AUC | Gini |
|---|---:|---:|
| Logistic Regression | 0.6276 | 0.2553 |
| Linear SVM | 0.6277 | 0.2554 |
| XGBoost | 0.6398 | 0.2795 |
| Weighted Average Ensemble | **0.6404** | **0.2809** |

The Weighted Average Ensemble achieved the highest ROC-AUC among the results presented above, although the improvement over XGBoost was relatively small.

---

## 7. Visualizations and Key Findings

### 7.1 Target Class Distribution

![Target Class Distribution](figures/target_distribution.png)

**Key finding:** The dataset exhibits a substantial class imbalance, making accuracy alone insufficient for evaluating classification performance.

### 7.2 Missing Value Analysis

![Missing Value Analysis](figures/missing_values.png)

**Key finding:** Missing-value patterns must be considered during preprocessing, as several anonymized features contain incomplete observations.

### 7.3 Model Performance Comparison

![Model Comparison](figures/model_comparison.png)

**Key finding:** XGBoost outperformed the linear classifiers in the reported experiments, suggesting that nonlinear relationships and feature interactions are relevant to the prediction task.

### 7.4 Ensemble Learning Comparison

![Ensemble Comparison](figures/ensemble_comparison.png)

**Key finding:** Weighted averaging provided a small improvement over the strongest individual model, while the benefits of ensemble learning depended on the combination strategy.

---

## 8. Technologies and Tools

**Programming:** Python

**Data Analysis:** pandas, NumPy

**Machine Learning:** scikit-learn, XGBoost

**Visualization:** Matplotlib, Seaborn

**Development:** Jupyter Notebook

**Methods:** Cross-Validation, Hyperparameter Optimization, Feature Engineering, Ensemble Learning

---

## 9. Full Project Report

A detailed description of the methodology, experiments, results, and conclusions is available in the complete Advanced Data Analytics E-Portfolio.

📄 [View the Full Project Report (PDF)](reports/ADA_E_Portfolio.pdf)

---

## 10. Conclusions

The experiments highlight several important findings:

- Logistic Regression and Linear SVM provided competitive linear baselines.
- XGBoost achieved stronger predictive performance than the investigated linear models.
- Hyperparameter optimization and preprocessing choices influenced model performance.
- Weighted averaging produced a modest improvement over the best individual classifier.
- Class imbalance and anonymized features remain important challenges for insurance claim prediction.

Overall, the project demonstrates an end-to-end machine learning workflow, from exploratory data analysis to model optimization and ensemble evaluation.

