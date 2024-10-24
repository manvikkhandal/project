```markdown
# Calorie Burnt Prediction using Machine Learning

This project predicts the number of calories burnt based on various input features using machine learning techniques. The model is built in Python, utilizing popular libraries like NumPy, Pandas, and Scikit-learn.

## Overview

The goal of this project is to create an efficient machine learning model that can predict the calories burnt during physical activities. By analyzing the given data, different regression models are trained and evaluated for performance.

## Libraries Used

- **NumPy**: For numerical operations
- **Pandas**: For data manipulation and analysis
- **Scikit-learn**: For machine learning model building and evaluation

## Models Implemented

1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**

## Installation

To run this project, clone the repository and install the required libraries using:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset contains features related to physical activity that influence calorie burning. You can load your dataset using Pandas.

```python
import pandas as pd

# Load your dataset
data = pd.read_csv('your_dataset.csv')
```

## Model Training

We split the dataset into training and testing sets using `train_test_split`. Multiple machine learning algorithms are implemented and compared for their prediction performance.

```python
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Linear Regression

```python
from sklearn.linear_model import LinearRegression

# Initialize and train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
```

### Decision Tree Regressor

```python
from sklearn.tree import DecisionTreeRegressor

# Initialize and train the model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
```

### Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

# Initialize and train the model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
```

## Results

After training the models, we evaluate them based on their accuracy and performance metrics such as R-squared and Mean Squared Error (MSE).

## How to Use

1. Clone the repository:
    ```bash
    git clone https://github.com/manvikkhandal/calorie-burnt-prediction.git
    ```
2. Install the required libraries.
3. Run the Python script to train the model and make predictions.
