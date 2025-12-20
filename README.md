# Task 1: Data Analysis and Preprocessing  
**Improved Detection of Fraud Cases for E-commerce and Bank Transactions**

## Overview  
Task 1 focuses on preparing high-quality, feature-rich datasets for fraud detection modeling. The objective is to clean, explore, enrich, and transform both e-commerce and bank transaction data while addressing the critical challenge of severe class imbalance.  

This task establishes a strong foundation for machine learning modeling, explainability, and business-driven evaluation in later stages of the project.

## Business Context  
Adey Innovations Inc. operates in the financial technology sector, where fraud detection accuracy directly affects:  
- Financial loss prevention  
- Customer trust and user experience  
- Transaction security and monitoring efficiency  

Inadequate preprocessing can result in high false positives that frustrate users or false negatives that lead to direct financial losses. Therefore, this task emphasizes data integrity, meaningful feature engineering, and imbalance-aware preparation.

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

## Project Structure ( Task 1)  
```bash
data/
├── raw/
│   ├── Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│   └── creditcard.csv
│
└── processed/
    ├── fraud_data_processed.csv
    └── creditcard_processed.csv

notebooks/
├── eda-fraud-data.ipynb
├── eda-creditcard.ipynb
├── feature-engineering.ipynb
