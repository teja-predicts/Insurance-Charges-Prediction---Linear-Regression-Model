from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns  # To improve plot aesthetics

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)
print(df.head())  # Preview the data

# Step 2: Preprocess the data (one-hot encoding for categorical variables)
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Step 3: Prepare the data for the model
X = df.drop(columns=['charges'])  # Features (independent variables)
y = df['charges']  # Target variable (dependent variable)

# Step 4: Split the data into training and test sets (optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set (better practice than using all data)
predictions = model.predict(X_test)

# Step 7: Evaluate the model
print(f"R-squared (on test set): {model.score(X_test, y_test)}")

# Step 8: Visualize the predictions

# Create a scatter plot to compare actual vs predicted charges
plt.figure(figsize=(8,6))
sns.set(style="whitegrid")  # Use a white grid background for better readability

# Scatter plot for actual vs predicted
plt.scatter(y_test, predictions, color='blue', alpha=0.6, edgecolors="w", s=100)
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', lw=2, linestyle='--')  # 45-degree line (ideal model)

# Adding labels and title
plt.xlabel('Actual Charges', fontsize=12)
plt.ylabel('Predicted Charges', fontsize=12)
plt.title('Actual vs Predicted Charges - Linear Regression', fontsize=14)

# Show the plot
plt.show()

# Bonus: Display a residual plot (difference between actual and predicted values)
plt.figure(figsize=(8,6))
residuals = y_test - predictions
sns.residplot(predictions, residuals, lowess=True, color="g", line_kws={'color': 'red', 'lw': 2})
plt.xlabel('Predicted Charges', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residuals Plot', fontsize=14)
plt.show()
