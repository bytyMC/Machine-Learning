import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/heart_disease.csv')
# Print the head of the data and it's description (Optional)
# print(data.head())
# print(data.describe().T)
print("Missing values in each column:")
print(data.isnull().any()) # We don"t have null values => we won"t remove anything

# Visualize the target distribution
data['target'].value_counts().plot(kind='bar', title='Distribution of Heart Disease')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Separate the features from the label (target)
X = data.iloc[:,:13].values
y = data["target"].values

# Split the data: 70/30
X_train,X_test,y_train, y_test = train_test_split(X, y, test_size = 0.3 , random_state = 0)

# Scale the data: Learn parameters from training set, then apply the learned parameters on the test set
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Add layers to DL model
classifier = Sequential()
classifier.add(Dense(activation = "relu", input_dim = 13, units = 8, kernel_initializer = "uniform"))
classifier.add(Dense(activation = "relu", units = 14, kernel_initializer = "uniform"))
classifier.add(Dense(activation = "sigmoid", units = 1, kernel_initializer = "uniform"))

# Compile the model
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics = ['accuracy'] )

# Train the model and store history
history = classifier.fit(X_train, y_train, batch_size = 8, epochs = 100, validation_split=0.2)

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Visualization.png")

# Make prediction
y_pred = classifier.predict(X_test)
# Convert to binary class labels: If higher than 0.5: true (has disease), else: false (no disease)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)

# Compute accuracy based on confusion matrix
accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred))