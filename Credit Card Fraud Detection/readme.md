# Credit Card Fraud Detection Using Machine Learning

## Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. With the rise of online transactions, detecting fraud has become critical to maintaining trust in the financial system. In this project, we leverage machine learning models to accurately classify transactions as fraudulent or legitimate based on various transaction features.

## Project Structure

The repository is organized as follows:

```
├── data/                # Dataset files (credit card transaction data)
├── notebooks/           # Jupyter notebooks for analysis and model building
├── models/              # Saved models in .joblib or .pkl format
├── scripts/             # Python scripts for data processing and model training
├── README.md            # Project documentation
└── requirements.txt     # List of dependencies
```

## Dataset

The dataset used for this project contains transactions made by credit card holders in September 2013. It includes a mix of fraudulent and legitimate transactions. Each transaction is described by a set of features, which have been anonymized due to privacy concerns.

- **Number of records**: 284,807
- **Number of fraudulent transactions**: 492
- **Number of legitimate transactions**: 284,315

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download).

## Approach

1. **Data Preprocessing**:
    - Handle missing values and outliers.
    - Normalize the data to improve model performance.
    - Address class imbalance through techniques such as SMOTE (Synthetic Minority Over-sampling Technique).
    
2. **Exploratory Data Analysis**:
    - Analyze the distribution of transactions.
    - Visualize the correlation between different features.

3. **Model Building**:
    - We implemented various machine learning models, including:
      - Logistic Regression
      - Decision Tree
      - Random Forest
      - Gradient Boosting
      - Support Vector Machine
    - The models are evaluated based on metrics such as accuracy, precision, recall, and the F1-score.
    
4. **Evaluation**:
    - We use a confusion matrix, AUC-ROC curve, and precision-recall curve to evaluate model performance.
    - Cross-validation is performed to ensure the model's robustness.

## Dependencies

To run the project, you will need the following dependencies:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Imbalanced-learn (for handling imbalanced data)
- Jupyter (optional, for running notebooks)

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/manvikkhandal/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Run the Jupyter notebooks in the `notebooks/` folder to explore the data, train models, and evaluate results.
3. For a production-ready model, use the scripts in the `scripts/` folder to train a model on the full dataset and save it for later use.
