import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("data/sales_data.csv")

# Check for missing values
print("Missing values in each column:\n", data.isnull().sum())

# Check correlation between features
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.savefig("correlation.png")

# Define features and labels
X = np.array(data.drop(["Sales"], axis=1))  # Corrected 'drop' axis
y = np.array(data["Sales"])

# Split data into train and test sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit Linear Regression model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Predict on test set
ypred = model.predict(xtest)

# Combine predictions with actual sales for comparison
results = pd.DataFrame({
    "Actual Sales": ytest.flatten(),
    "Predicted Sales": ypred.flatten()
})

print("Actual vs Predicted Sales:\n", results)

plt.figure(figsize=(10, 6))
plt.scatter(ytest, ypred, color='blue', label='Predicted vs Actual')
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red', linestyle='--', label='Perfect Fit')  # Ideal 1:1 line
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.savefig("actual_vs_predicted.png")

results.to_csv("data/sales_predictions.csv", index=False)
