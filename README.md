# Heart Disease Risk Analysis & Prediction

## 1. Project Overview

This project analyzes personal health indicators from the 2022 CDC BRFSS survey to identify risk factors for heart disease. By applying data science and machine learning techniques, the team aims to detect patterns in lifestyle and health factors (such as sleep, mental health, and physical activity) that predict a patient's condition.

The project emphasizes not just prediction accuracy but also **interpretability**, the **real-world cost** of false diagnoses (Precision-Recall trade-off), and the identification of latent patient subgroups.

## 2. Team Information

| Student ID | Name | Role |
| --- | --- | --- |
| 23127130 | **Nguyễn Hữu Anh Trí** | Data Analysis & Modeling |
| 23127051 | **Cao Tấn Hoàng Huy** | Data Analysis & Modeling |

## 3. Dataset Source & Description

* **Source**: [CDC 2022 Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) (via Kaggle).
* **Original Source**: Centers for Disease Control and Prevention (CDC) - Behavioral Risk Factor Surveillance System (BRFSS).
* **Description**: The dataset represents the health status of the U.S. adult population (2022). It includes **~445,000 records** with **40 features** covering:
* **Demographics**: Age, Sex, Race, State.
* **Health Metrics**: BMI, Physical/Mental Health Days, General Health status.
* **Conditions**: Heart Attack, Stroke, Asthma, Diabetes, Kidney Disease.
* **Lifestyle**: Smoking, Alcohol, Physical Activity, Sleep Hours.


* **File Used**: `heart_2022_with_nans.csv` (Preserves missing values for proper handling).

## 4. Research Questions

### **Question 1: The Mind–Body connection and sleep**

**The Question** > *"How do mental health (`MentalHealthDays`) and sleep duration (`SleepHours`) interact in relation to Heart Attack risk? Does physical activity (`PhysicalActivities`) act as a protective factor that mitigates cardiovascular risk in individuals with poor sleep or psychological stress?"*

### **Question 2: The Compounding Effect of Multimorbidity**

**The Question** > *"How does the risk of Heart Attack (`HadHeartAttack`) change as an individual suffers from multiple chronic conditions simultaneously (Diabetes, Kidney Disease, COPD, Asthma, etc.)? Is this disease burden more impactful in younger age groups (lower `AgeCategory`) compared to older ones?"*
---

### **Question 3: Patient Profile Clustering (Unsupervised Learning – Clustering)**

**The Question** > *"Can we segment the surveyed population into distinct 'Health Profiles' based on a comprehensive set of demographic, physical, mental, and lifestyle factors? Which cluster has the highest Heart Attack risk, and what underlying patterns distinguish high-risk from low-risk groups?"*

### **Question 4: Risk Prediction (Supervised Learning – Classification)**

**The Question** > *"Can we build a classification model to accurately predict whether an individual has experienced a Heart Attack using non-invasive screening features? Which model (Logistic Regression vs. Random Forest vs K-Nearest Neighbors) better balances predictive performance and interpretability?"*

## 5. Key Findings Summary

* **BMI is Insufficient**: The analysis reveals that relying on BMI alone is misleading, as there are many cases of "normal weight" individuals suffering heart attacks.
* **Strongest Predictors**: Variables like `AgeCategory`, `Sex`, and `GeneralHealth` demonstrated stronger predictive power than BMI.
* **Patient Archetypes**: Risk is not uniformly distributed. Patients clump into distinct profiles (phenotypes), suggesting that tailored interventions are more effective than one-size-fits-all approaches.
* **Modeling Trade-offs**: In medical screening, **Recall** (catching all positive cases) is often prioritized over Accuracy. The team found that optimizing for this trade-off (e.g., handling class imbalance) was critical, even if it meant more false alarms.

## 6. File Structure

* **`Data_Explore.ipynb`**:
* **Data Collection**: Downloads data using `kagglehub`.
* **Data Integrity**: Checks for duplicates (removed ~157 rows) and missing values (~44% of rows have missing data).
* **Exploration**: Detailed column inventory and definitions.


* **`Data_modeling.ipynb`**:
* **Preprocessing**: Implements **Min-Max Capping** (Winsorization) for outliers in `MentalHealthDays`, `SleepHours`, and `BMI`.
* **Analysis**: Deep dive into the research questions (Mind-Body connection, Multimorbidity, and Clustering).
* **Machine Learning**: Implementation of pipelines using Logistic Regression, Random Forest, and KNN with a focus on hyperparameter tuning and handling class imbalance (`class_weight`, `RandomUnderSampler`).



## 7. Dependencies

The project includes a `requirements.txt` file listing all necessary packages. To install them, run:

```bash
pip install -r requirements.txt

```

**Key libraries used:**

* **Core**: `pandas`, `numpy`
* **Visualization**: `matplotlib`, `seaborn`
* **ML & Preprocessing**: `sklearn` (Scikit-Learn), `imblearn` (Imbalanced-Learn)
* **Data Source**: `kagglehub`

## 8. How to Run

1. **Clone the repository** (or download the notebooks).
2. **Install Dependencies**: Run the `pip install -r requirements.txt` command.
3. **Run `Data_Explore.ipynb**`:
* This notebook will automatically download the dataset from Kaggle into a folder named `./my_heart_disease_data`.
* It performs initial cleaning and exploration.
* After that, it will create a new dataset named `cleaned_heart_data.csv`


4. **Run `Data_modeling.ipynb**`:
* Ensure the cleaned data is available (or the notebook points to the correct file path).
* Run all cells to reproduce the outlier handling, analysis, clustering, and model training.



---
