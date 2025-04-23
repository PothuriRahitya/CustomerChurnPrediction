# Customer Churn Prediction using Machine Learning (Telecom Industry)

This project presents an intelligent churn prediction system that leverages Machine Learning to identify customers likely to discontinue telecom services. By integrating multiple algorithms through ensemble learning, the system significantly improves prediction accuracy and supports proactive customer retention strategies.

## Project Overview

- Customer churn presents a major challenge in the telecom sector, directly affecting revenue and growth.
- Traditional churn detection methods often lack precision in identifying potential churners.
- This project addresses these gaps by:
  - Utilizing a large, comprehensive telecom dataset
  - Comparing traditional models like Logistic Regression and KNN
  - Implementing an ensemble voting classifier combining Logistic Regression, Random Forest, LightGBM, and KNN
  - Providing an interactive Streamlit web interface for real-time churn prediction

## Objectives

- Improve the accuracy and stability of churn predictions using ensemble techniques
- Minimize false negatives to retain valuable customers
- Provide a user-friendly interface for business users to interact with the model
- Enable scalable deployment across various customer datasets

## Methodology

- **Dataset**: Contains customer demographics, usage, billing, and subscription features
- **Preprocessing**:
  - Handling missing values
  - Encoding categorical variables
  - Feature scaling and normalization
- **Model Training**:
  - Algorithms: Logistic Regression, Random Forest, LightGBM, KNN
  - Ensemble: Hard Voting Classifier for combined predictions
- **Evaluation**:
  - Metrics: Accuracy, Precision, Recall, F1-Score, AUC
  - Cross-validation for robust evaluation

## Technologies Used

- Python
- Scikit-learn
- Pandas, NumPy
- Streamlit (web interface)
- Joblib (model serialization)
- Matplotlib and Seaborn (visualizations)
- VS Code, Anaconda

## Results Summary

| Model              | Accuracy | AUC    | Remarks                        |
|-------------------|----------|--------|--------------------------------|
| Logistic Regression | 98.11%   | 98.10% | Good baseline model            |
| KNN                 | 98.79%   | 98.84% | Strong local pattern detection |
| Random Forest       | 99.17%   | 99.04% | High overall accuracy          |
| LightGBM            | 97.87%   | 98.12% | Balanced and fast              |
| Voting Classifier   | 98.69%   | 98.80% | Most stable and accurate       |

## Web Interface

- Built using Streamlit
- Allows users to upload a CSV file containing customer data
- View real-time churn predictions
- Input individual customer details for instant results

## Future Enhancements

- Extend to other industries (banking, retail, media)
- Enable real-time predictions from live telecom feeds
- Integrate explainable AI for transparency in churn reasoning
- Explore deep learning models for further accuracy gains

## Authors

- P. Rahitya Sri Trilokya
- P. Gnana Pravallika
- P. Sharook Khan
- P. Rama Ganesh Sai Roopak
- **Guide**: Dr. P. Sudhakar, Professor, Dept. of CSE, VVIT

## Project Setup & Usage

## Project Setup & Usage

### 1. Clone the repository, install dependencies, and run the app

> **Note**: Make sure you have [Anaconda](https://www.anaconda.com/) installed on your system.  
> This project is designed to run within an **Anaconda environment**. Use the **Anaconda Prompt** to execute the commands below.

git clone https://github.com/PothuriRahitya/CustomerChurnPrediction.git
cd customer-churn-prediction-telecom

pip install --upgrade pip
pip install matplotlib numpy lightgbm streamlit scikit-learn

streamlit run app.py
