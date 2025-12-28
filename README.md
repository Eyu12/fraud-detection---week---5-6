# Task 1: Data Analysis and Preprocessing  
**Improved Detection of Fraud Cases for E-commerce and Bank Transactions**

## Overview  
Task 1 focuses on preparing high-quality, feature-rich datasets for fraud detection modeling. The objective is to clean, explore, enrich, and transform both e-commerce and bank transaction data while addressing the critical challenge of severe class imbalance.  

This task establishes a strong foundation for machine learning modeling, explainability, and business-driven evaluation in later stages of the project.

---

## Business Context  
Adey Innovations Inc. operates in the financial technology sector, where fraud detection accuracy directly affects:  
- Financial loss prevention  
- Customer trust and user experience  
- Transaction security and monitoring efficiency  

Inadequate preprocessing can result in high false positives that frustrate users or false negatives that lead to direct financial losses. Therefore, this task emphasizes data integrity, meaningful feature engineering, and imbalance-aware preparation.

---

## Datasets Used  

1. **Fraud_Data.csv (E-commerce Transactions)**  
   - Contains user-level and transaction-level information, including:  
     - User behavior: `signup_time`, `purchase_time`  
     - Transaction attributes: `purchase_value`, `device_id`  
     - Demographics: `age`, `sex`  
     - Network data: `ip_address`  
   - Target variable: `class`  
     - `1` → Fraudulent transaction  
     - `0` → Legitimate transaction  
   - Primary challenge: Highly imbalanced class distribution.

2. **IpAddress_to_Country.csv**  
   - Maps IP address ranges to countries using:  
     - `lower_bound_ip_address`  
     - `upper_bound_ip_address`  
     - `country`  
   - Used to enrich e-commerce transactions with geolocation information.

3. **creditcard.csv (Bank Transactions)**  
   - Includes:  
     - `Time`: Seconds elapsed since the first transaction  
     - `V1–V28`: Anonymized PCA-transformed features  
     - `Amount`: Transaction value  
     - `Class`: Target variable  
   - Primary challenge: Extreme class imbalance, typical of real-world bank fraud data.

## Task 1 Objectives  
- Clean and validate raw transaction data  
- Perform exploratory data analysis (EDA)  
- Engineer meaningful behavioral and time-based features  
- Integrate geolocation data using IP range matching  
- Transform features for machine learning readiness  
- Handle class imbalance using appropriate resampling techniques

---

## Methodology  

1. **Data Cleaning**  
   - Converted timestamp fields to datetime format  
   - Removed duplicate records  
   - Imputed missing values using:  
     - Median for numerical features  
     - Mode for categorical features  
   - Corrected data types and validated IP formats  

2. **Exploratory Data Analysis (EDA)**  
   - Univariate analysis: Distribution of key variables  
   - Bivariate analysis: Relationship between features and fraud labels  
   - Class distribution analysis: Quantification of fraud imbalance  
   - Visualizations used: histograms, boxplots, bar charts  

3. **Geolocation Integration**  
   - Converted IP addresses to integer format  
   - Performed range-based matching between transaction IPs and country ranges  
   - Analyzed fraud patterns by country  
   - Revealed strong geographic signals associated with fraudulent behavior  

4. **Feature Engineering (Fraud_Data.csv)**  
   - New features created:  
     - Hour of day of transaction  
     - Day of week of transaction  
     - Time elapsed between signup and purchase  
     - Transaction velocity within defined time windows  
     - Aggregated user behavior statistics  
   - Features capture behavioral patterns commonly associated with fraud  

5. **Data Transformation**  
   - Numerical features scaled using `StandardScaler`  
   - Categorical features encoded using `One-Hot Encoding`  
   - Data leakage avoided by applying transformations only to training data  

6. **Handling Class Imbalance**  
   - SMOTE (Synthetic Minority Oversampling Technique) applied to training data only  
   - Class distributions documented before and after resampling  
   - SMOTE chosen to preserve minority class information while maintaining sufficient legitimate transaction data  

---

## Outputs  
- Cleaned and feature-engineered datasets saved in `data/processed/`  
- Exploratory visualizations and statistical summaries  
- Model-ready datasets prepared for fraud detection modeling  

## Key Takeaways  
- Fraud detection requires careful, imbalance-aware preprocessing  
- Behavioral and geolocation features significantly enhance fraud detection  
- Proper data preparation reduces false positives and missed fraud cases  
- High-quality preprocessing directly supports trustworthy and explainable models  

## Next Steps  
- **Task 2:** Model training and evaluation using imbalance-aware metrics  
- **Task 3:** Model explainability using SHAP and business interpretation  
- Model deployment considerations for real-time fraud detection systems

---

# Task 2 – Model Building and Training

## Objective
The objective of this task is to build, train, and evaluate classification models for detecting fraudulent transactions. Special attention is given to **imbalanced datasets** and selecting an optimal model based on both **performance** and **interpretability**.

---

## Datasets
Two datasets are used:

- **creditcard.csv**  
  - Target variable: `Class`  
  - `0` → Non-fraudulent transaction  
  - `1` → Fraudulent transaction  

- **Fraud_Data.csv**  
  - Target variable: `class`  
  - `0` → Legitimate transaction  
  - `1` → Fraudulent transaction  

---

## Data Preparation
1. **Feature–Target Separation**  
   Independent variables (`X`) are separated from the target variable (`y`).

2. **Stratified Train-Test Split**  
   Data is split into training and testing sets using a **stratified split** to preserve class distribution.  
   Default split ratio: 80% training, 20% testing.

3. **Imbalanced Data Handling**  
   Techniques such as **SMOTE** or class-weighted learning are applied where appropriate.

---

## Baseline Model: Logistic Regression
A Logistic Regression model is trained as a **baseline** due to its simplicity and interpretability.

**Evaluation Metrics**:
- AUC-PR (Area Under Precision–Recall Curve)  
- F1-Score  
- Confusion Matrix  

---

## Ensemble Model
An ensemble-based classifier is trained to improve predictive performance.

**Model Options**:
- Random Forest  
- XGBoost  
- LightGBM  

**Hyperparameter Tuning**:
- `n_estimators`  
- `max_depth`  
- Other model-specific parameters as needed  

**Evaluation Metrics** (same as baseline):
- AUC-PR  
- F1-Score  
- Confusion Matrix  

---

## Cross-Validation
- **Stratified K-Fold Cross-Validation** (k = 5)  
- Preserves class distribution in each fold  

**Reported Results**:
- Mean performance across folds  
- Standard deviation of metrics  

---

## Model Comparison and Selection
All trained models are compared side-by-side using:

- AUC-PR  
- F1-Score  
- Confusion Matrix  
- Cross-validation statistics (mean ± standard deviation)  

**Model Selection Criteria**:
- Predictive performance  
- Stability across folds  
- Interpretability and deployment suitability  

A clear justification is provided for the chosen model.

---

## Outputs
- Trained models  
- Evaluation metrics and plots  
- Cross-validation results  
- Final model selection summary  

---

## Conclusion
This task demonstrates a structured approach to fraud detection modeling, balancing **performance**, **robustness**, and **interpretability**, while addressing challenges posed by highly imbalanced datasets.

# Task 3 – Model Explainability

## Objective

Interpret the predictions of the best-performing fraud detection model using **SHAP (SHapley Additive exPlanations)** to understand the key drivers of fraud and translate insights into actionable business recommendations.

---

## Overview

This task focuses on model transparency and interpretability through:

1. Baseline feature importance analysis  
2. SHAP-based global and local explanations  
3. Interpretation of results  
4. Business-focused recommendations  

---

## 1. Feature Importance Baseline

### Purpose
To obtain an initial understanding of feature influence using the model’s built-in feature importance.

### Steps
- Extract feature importance from the trained ensemble model (e.g., Random Forest, XGBoost).
- Rank features by importance score.
- Visualize the **top 10 most important features** using a bar chart.

### Output
- Feature importance values
- Bar plot of top 10 features

---

## 2. SHAP Analysis

### Why SHAP?
SHAP provides:
- Global interpretability (overall feature importance)
- Local interpretability (individual prediction explanations)
- Consistent and theoretically grounded explanations

---

### 2.1 SHAP Summary Plot (Global)

#### Objective
Identify the most influential features across all predictions.

#### Deliverable
- SHAP summary (beeswarm) plot showing:
  - Feature importance ranking
  - Direction of impact (positive or negative)
  - Distribution of SHAP values

---

### 2.2 SHAP Force Plots (Local)

#### Objective
Explain individual transaction-level predictions.

#### Required Cases
Generate SHAP force plots for:

1. **True Positive (TP)**  
   - Fraud correctly detected

2. **False Positive (FP)**  
   - Legitimate transaction incorrectly flagged as fraud

3. **False Negative (FN)**  
   - Fraud transaction missed by the model

#### Focus
- Features pushing the prediction toward fraud
- Features pushing the prediction toward non-fraud
- Final decision rationale

---

## 3. Interpretation

### 3.1 Feature Importance Comparison
- Compare built-in model feature importance with SHAP global importance.
- Identify overlapping and divergent influential features.

### 3.2 Key Fraud Drivers
- Identify the **top 5 features** driving fraud predictions based on SHAP values.
- Explain:
  - Direction of impact
  - Magnitude of influence
  - Business or domain relevance

### 3.3 Surprising or Counterintuitive Findings
- Features expected to be important but showing low SHAP impact
- Legitimate behavior patterns contributing to false positives
- Fraud cases missed due to feature interactions

---

## 4. Business Recommendations

### Objective
Translate explainability insights into **actionable fraud prevention strategies**.

### Requirements
- Provide at least **three recommendations**
- Each recommendation must:
  - Be specific and actionable
  - Be supported by SHAP insights
  - Address fraud risk or operational efficiency

### Example Recommendations

1. **Enhanced Verification for New Accounts**  
   Transactions occurring shortly after account creation show high SHAP impact.  
   *Recommendation:* Apply additional verification for transactions within the first X hours of signup.

2. **Dynamic Controls for High-Value Transactions**  
   Transaction amount is a strong fraud driver.  
   *Recommendation:* Use adaptive risk thresholds for high-value transactions.

3. **Manual Review for Borderline Cases**  
   False positives often show mixed SHAP signals.  
   *Recommendation:* Route transactions with moderate SHAP risk scores to manual review.

---

## Conclusion

This task demonstrates how SHAP improves model transparency by:
- Explaining global and local prediction behavior
- Validating model decision logic
- Identifying causes of false positives and false negatives
- Enabling data-driven and explainable business decisions


## Project Structure 
```bash
# Create the complete project structure
fraud-detection/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/                           # Add this folder to .gitignore
│   ├── raw/                      # Original datasets
│   └── processed/         # Cleaned and feature-engineered data
├── notebooks/
│   ├── __init__.py
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   ├── shap-explainability.ipynb
│   └── README.md
├── src/
│   ├── __init__.py
├── tests/
│   ├── __init__.py
├── models/                      # Saved model artifacts
├── scripts/
│   ├── __init__.py
│   └── README.md
├── requirements.txt
├── README.md
└── .gitignore