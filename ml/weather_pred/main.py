import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("data/weather-data.csv")

# Print head and info of the data (Optional) => we need to know how many null values are present
# print(data.head())
# print(data.info())
# print(data.isnull().sum())

# Convert the "date" column to date datatype
data['date'] = pd.to_datetime(data['date'])

# Plot the data // The plot can be visualized via plt.show()
plt.figure(figsize=(10,5))
sns.set_theme()
sns.countplot(x = 'weather',data = data,palette="ch:start=.2,rot=-.3")
plt.xlabel("weather",fontweight='bold',size=13)
plt.ylabel("Count",fontweight='bold',size=13)
plt.savefig("weather_countplot.png")

# Plot date / max temperature
fig1 = px.line(data_frame = data,
       x = 'date',
       y = 'temp_max', 
       title = 'Variation of Maximum Temperature')

fig1.show()

plt.figure(figsize=(10,5))
sns.catplot(x='weather',y ='temp_max',data=data,palette="crest")
plt.savefig("weather_crest.png")

# Plot date / min temperature
fig3 = px.line(data_frame = data,
       x = 'date',
       y = 'temp_min', 
       title = 'Variation of Minimum Temperature')

fig3.show()

plt.figure(figsize=(10,5))
sns.catplot(x='weather',y ='temp_min',data=data,palette = "RdBu")
plt.savefig("weather_RdBu.png")

label_encoder = preprocessing.LabelEncoder()
data["weather"] = label_encoder.fit_transform(data["weather"])
data["weather"].unique()

# Date column has no importance in the prediction => remove
data = data.drop('date',axis=1)

# Get features and label
X = data.drop('weather', axis=1)
y = data['weather']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Data standardization // learn parameter from training set, then apply on the test set
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the model - Logistic Regression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predict
y_pred = classifier.predict(X_test)

# Evalutate the predictions
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True, fmt = '.3g')
acc1 = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {acc1}") # 0.76
# Save the cm (Optional)
# plt.savefig("weather_pred_eval_lr.png")

# Train second model - GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Evaluate predictions
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot = True, fmt = '.3g')
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy score : {acc}") # 0.84
# Save the cm (Optional)
# plt.savefig("weather_pred_eval_gnb.png")

# Train third model - SVM
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot = True, fmt = '0.3g')
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy score : {acc}") # 0.79
# Save the cm (Optional)
# plt.savefig("weather_pred_eval_svm.png")