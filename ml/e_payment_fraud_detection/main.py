import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle

data = pd.read_csv("data/credit_card.csv")

# Check if the correct file is loaded (Opionally)
# print(data.head())

type = data["type"].value_counts()

# Extract the transaction's names
transactions = type.index
# Extract the quantity of each transaction
quantity = type.values

# See the distribution of the transactions in a pie chart
figure = px.pie(data, 
             values=quantity, 
             names=transactions, hole = 0.5, 
             title="Distribution of Transaction Type")

# Show chart (Optionally)
# figure.show()

# Check correlation
correlation = data.corr()
# Print (Optional)
# print(correlation["isFraud"].sort_values(ascending=False))

# Preprocessing
# Transform the categorical features into numerical 
# type column => integer 
# isFraud column => No Fraud / Fraud labels

data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})

# Print to see the data (Optional)
# print(data.head())

# Split the data in features and label
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

# Train the model using DT (it may take a while)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

# Save the model, so you don't have to train it everytime
# with open("dt_model_fraud_detetion.pkl", "wb") as f:
#    pickle.dump(model,f)

# Load the saved model
# with open("dt_model_fraud_detetion.pkl", "rb") as f:
#    model = pickle.load(f)   

# Print the accuracy of the model (Optional)
print(model.score(xtest, ytest)) # 0.999

# Make a prediction
# features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]]) # New sample
ypred = model.predict(features)

print(ypred)

# More evaluation markers: precision, recall, F1-score, ROC-AUC
print(classification_report(ytest, ypred))
print("ROC-AUC Score:", roc_auc_score(ytest, model.predict_proba(xtest)[:, 1]))
