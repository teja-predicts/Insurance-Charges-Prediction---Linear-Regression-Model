from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)

# Step 2: Preprocess the data (one-hot encoding for categorical variables)
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Step 3: Prepare the data for the model
X = df.drop(columns=['charges'])  # Features (independent variables)
y = df['charges']  # Target variable (dependent variable)

# Step 4: Initialize the Linear Regression model
model = LinearRegression()

# Step 5: Perform 5-fold cross-validation and compute Negative Mean Squared Error (MSE)
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Step 6: Print the cross-validation scores (Negative MSE)
print("Cross-validation scores (Negative Mean Squared Error):", scores)

# Step 7: Calculate and print the mean and standard deviation of the cross-validation scores
print("Mean score (Negative MSE):", scores.mean())
print("Standard deviation of scores:", scores.std())

# Step 8: Train the model on the full dataset and make predictions
model.fit(X, y)
predictions = model.predict(X)

# Step 9: Visualize the predictions

# Create a scatter plot to compare actual vs predicted charges
plt.figure(figsize=(8,6))
sns.set(style="whitegrid")  # Use a white grid background for better readability

# Scatter plot for actual vs predicted
plt.scatter(y, predictions, color='blue', alpha=0.6, edgecolors="w", s=100)

# Ideal line (45-degree line) where predicted equals actual
plt.plot([0, max(y)], [0, max(y)], color='red', lw=2, linestyle='--')

# Adding labels and title
plt.xlabel('Actual Charges', fontsize=12)
plt.ylabel('Predicted Charges', fontsize=12)
plt.title('Actual vs Predicted Charges - Linear Regression', fontsize=14)

plt.show()

# Step 10: Bonus: Display a residual plot (difference between actual and predicted values)
residuals = y - predictions

plt.figure(figsize=(8,6))
sns.residplot(predictions, residuals, lowess=True, color="g", line_kws={'color': 'red', 'lw': 2})
plt.xlabel('Predicted Charges', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residuals Plot', fontsize=14)
plt.show()