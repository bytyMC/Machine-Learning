import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Load data
data = pd.read_csv('data/wine_data.csv')

# Check for null values
print(data.isnull().any())

# Impute missing values with mean for numerical columns
for col in data.columns:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].mean())

# Verify if any null values remain
print("Total missing values left:", data.isnull().sum().sum())

# Visualize the distribution of the data
data.hist(bins=20, figsize=(10,10))
plt.savefig("histograms.png")

# Visualize relationship between quality and alcohol
plt.figure(figsize=(10, 6))
plt.bar(data['quality'], data['alcohol'])
plt.xlabel('Quality')
plt.ylabel('Alcohol')
plt.savefig("quality_alcohol.png")

# Heatmap for correlations > 0.7
plt.figure(figsize=(12, 12))
sb.heatmap(data.corr() > 0.7, annot=True, cbar=False)
plt.savefig("correlation.png")

# Drop highly correlated column: 'total sulfur dioxide' and 'free sulphur dioxide' columns
data = data.drop('total sulfur dioxide', axis=1)

# Feature Engineering: Create a binary target variable
data['best quality'] = [1 if x > 5 else 0 for x in data.quality]  
data.replace({'white': 1, 'red': 0}, inplace=True)

# Separate features and labels
features = data.drop(['quality', 'best quality'], axis=1)
label = data['best quality']

# Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(features, label, test_size=0.2, random_state=40)

# Impute missing values  using the mean strategy
imputer = SimpleImputer(strategy='mean')  
xtrain = imputer.fit_transform(xtrain)
xtest = imputer.transform(xtest)

# Normalize the data 
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)

# Initialize models
models = [LogisticRegression(max_iter=1000), XGBClassifier(use_label_encoder=False, eval_metric='logloss'), SVC(kernel='rbf')]

# Train models and evaluate using ROC-AUC score
for model in models:
    model.fit(xtrain, ytrain)
    ytrain_pred = model.predict(xtrain)
    ytest_pred = model.predict(xtest)

    print(f'{model.__class__.__name__}:')
    print('Training ROC-AUC Score: ', metrics.roc_auc_score(ytrain, ytrain_pred))
    print('Validation ROC-AUC Score: ', metrics.roc_auc_score(ytest, ytest_pred))
    print()

# Confusion Matrix for XGBoost
cm = confusion_matrix(ytest, models[1].predict(xtest))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]) 
disp.plot()
plt.savefig("conf_matrix.png")

# Classification report for XGBoost
print("Classification Report for XGBoost:\n", classification_report(ytest, models[1].predict(xtest)))

