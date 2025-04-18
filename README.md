# Insurance-Charges-Prediction---Linear-Regression-Model
This project builds a Linear Regression model to predict insurance charges based on demographic and lifestyle factors such as age, BMI, smoking status, and more.
The dataset comes from the "Machine Learning with R" datasets collection.

---

## Project Overview
- Load and explore the dataset
- Preprocess the data (handle categorical variables with one-hot encoding)
- Split the data into training and testing sets
- Train a Linear Regression model
- Evaluate model performance using **R² score**
- Visualize:
  - Actual vs Predicted Charges
  - Residuals plot to assess model errors

---

## Dataset
- **Source**: [Insurance.csv Dataset](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv)
- **Description**: Contains information about individuals, including:
  - Age
  - Sex
  - BMI
  - Number of children
  - Smoking status
  - Region
  - Insurance charges (target variable)

---

## Key Steps

### 1. Data Preprocessing
- Categorical variables (`sex`, `smoker`, `region`) are converted into numeric format using one-hot encoding.

### 2. Model Training
- Linear Regression model from `scikit-learn` is trained on 80% of the data.
- 20% of the data is reserved for testing.

### 3. Model Evaluation
- **R-squared Score** is reported on the test set to measure how well the model explains variance in insurance charges.

### 4. Visualization
- **Scatter Plot** comparing actual vs predicted insurance charges.
- **Residuals Plot** to check for model biases or patterns in errors.

---

## Libraries Used
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## How to Run
1. Clone this repository
2. Install the required libraries:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
