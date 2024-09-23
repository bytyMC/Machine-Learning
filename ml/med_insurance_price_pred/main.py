import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score
from pickle import dump
from feature_engine.outliers import ArbitraryOutlierCapper

# Load the dataset
data = pd.read_csv("data/med_insurance.csv")

# Check for missing values
print("Null values per column:\n", data.isnull().sum())

# Visualize categorical data (sex, smoker, region)
plt.figure(figsize=(20, 10))
features = ['sex', 'smoker', 'region']
for i, col in enumerate(features):
    plt.subplot(1, 3, i + 1)
    counts = data[col].value_counts()
    plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
plt.show()

# Visualize relationships between categorical features and charges
plt.figure(figsize=(20, 10))
features = ['sex', 'children', 'smoker', 'region']
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    data.groupby(col)['charges'].mean().plot.bar()
plt.title(f'{col} vs Charges')
plt.show()

# Scatter plots for age and bmi against charges
plt.figure(figsize=(17, 7))
for i, col in enumerate(['age', 'bmi']):
    plt.subplot(1, 2, i + 1)
    sns.scatterplot(data=data, x=col, y='charges', hue='smoker')
plt.show()

# Drop duplicates and check for outliers in 'bmi'
data.drop_duplicates(inplace=True)
sns.boxplot(data['bmi'])
plt.show()

# Outlier capping using IQR for 'bmi'
Q1, Q3 = data['bmi'].quantile([0.25, 0.75])
IQR = Q3 - Q1
low_lim, up_lim = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

# Apply outlier capper
arb = ArbitraryOutlierCapper(min_capping_dict={'bmi': low_lim}, max_capping_dict={'bmi': up_lim})
data[['bmi']] = arb.fit_transform(data[['bmi']])
sns.boxplot(data['bmi'])
plt.show()

# Encode categorical features
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data['region'] = data['region'].map({'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3})

# Data correlations (optional to visualize)
print("Correlation matrix:\n", data.corr())

# Split data into features and target
X = data.drop(['charges'], axis=1)
Y = data['charges']

# Split the dataset into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train and evaluate models
def evaluate_model(model, xtrain, xtest, ytrain, ytest):
    model.fit(xtrain, ytrain)
    ypred_train = model.predict(xtrain)
    ypred_test = model.predict(xtest)
    
    print(f"Train R2 Score: {r2_score(ytrain, ypred_train)}")
    print(f"Test R2 Score: {r2_score(ytest, ypred_test)}")
    print(f"Cross-validation score: {cross_val_score(model, X, Y, cv=5).mean()}\n")

# Linear Regression
print("Linear Regression:")
evaluate_model(LinearRegression(), xtrain, xtest, ytrain, ytest)

# Support Vector Regressor
print("SVR:")
evaluate_model(SVR(), xtrain, xtest, ytrain, ytest)

# Random Forest Regressor with Grid Search
rf_params = {'n_estimators': [10, 40, 50, 100, 120, 150]}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid=rf_params, scoring="r2", cv=5)
rf_grid.fit(xtrain, ytrain)
print("Best Random Forest parameters:", rf_grid.best_params_)

rf_best = RandomForestRegressor(random_state=42, n_estimators=rf_grid.best_params_['n_estimators'])
print("Random Forest Regressor:")
evaluate_model(rf_best, xtrain, xtest, ytrain, ytest)

# Gradient Boosting Regressor with Grid Search
gb_params = {'n_estimators': [10, 20, 50], 'learning_rate': [0.1, 0.2, 0.5]}
gb_grid = GridSearchCV(GradientBoostingRegressor(), param_grid=gb_params, scoring="r2", cv=5)
gb_grid.fit(xtrain, ytrain)
print("Best Gradient Boosting parameters:", gb_grid.best_params_)

gb_best = GradientBoostingRegressor(n_estimators=gb_grid.best_params_['n_estimators'], learning_rate=gb_grid.best_params_['learning_rate'])
print("Gradient Boosting Regressor:")
evaluate_model(gb_best, xtrain, xtest, ytrain, ytest)

# XGBoost Regressor with Grid Search
xgb_params = {'n_estimators': [10, 20, 40], 'max_depth': [3, 4, 5], 'gamma': [0, 0.15, 0.3]}
xgb_grid = GridSearchCV(XGBRegressor(), param_grid=xgb_params, scoring="r2", cv=5)
xgb_grid.fit(xtrain, ytrain)
print("Best XGBoost parameters:", xgb_grid.best_params_)

xgb_best = XGBRegressor(n_estimators=xgb_grid.best_params_['n_estimators'], max_depth=xgb_grid.best_params_['max_depth'], gamma=xgb_grid.best_params_['gamma'])
print("XGBoost Regressor:")
evaluate_model(xgb_best, xtrain, xtest, ytrain, ytest)

# Save the final model
dump(xgb_best, open('insurance_model.pkl', 'wb'))

# Test prediction with new data
new_data = pd.DataFrame({'age': 19, 'sex': 0, 'bmi': 27.9, 'children': 0, 'smoker': 1, 'region': 1}, index=[0])
predicted_charge = xgb_best.predict(new_data)
print(f"Predicted charge for new data: {predicted_charge[0]}")