import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn import metrics

DATA_PATH = 'data/lung_cancer/lung_colon_image_set/lung_image_sets'
IMG_SIZE = 256
SPLIT_RATIO = 0.2
EPOCHS = 10
BATCH_SIZE = 64

# Load and display sample images from each category
def display_sample_images(classes, path):
    for category in classes:
        image_dir = f'{path}/{category}'
        images = os.listdir(image_dir)

        # Create subplots for each category
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Sample Images for {category}', fontsize=20)

        # Display 3 random images from each category
        for i in range(3):
            img = np.array(Image.open(f'{image_dir}/{images[np.random.randint(0, len(images))]}'))
            ax[i].imshow(img)
            ax[i].axis('off')
        plt.show()

# Preprocess images: resize and store in arrays
def load_images(classes, path, img_size):
    X, Y = [], []
    for i, category in enumerate(classes):
        image_paths = glob(f'{path}/{category}/*.jpeg')
        print(f'Loading images for {category}... ({len(image_paths)} images)')

        for image_path in image_paths:
            img = cv2.imread(image_path)
            X.append(cv2.resize(img, (img_size, img_size)))
            Y.append(i)  # Append the class label
    return np.asarray(X), pd.get_dummies(Y).values  # One-hot encode Y

# Build the CNN model
def create_model(img_size):
    model = keras.models.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=(img_size, img_size, 3), padding='same'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(3, activation='softmax')  # 3 output categories
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Callback to stop training when validation accuracy exceeds 90%
class StopAt90Accuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > 0.90:
            print('\nReached 90% validation accuracy. Stopping training.')
            self.model.stop_training = True

# Train the model
def train_model(model, X_train, Y_train, X_val, Y_val, batch_size, epochs):
    early_stopping = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)

    # Train the model with callbacks
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=batch_size, epochs=epochs, verbose=1, 
                        callbacks=[early_stopping, reduce_lr, StopAt90Accuracy()])
    
    return history

# Plot training history: accuracy and loss
def plot_training_history(history):
    history_data = pd.DataFrame(history.history)
    
    # Plot loss
    history_data[['loss', 'val_loss']].plot(title='Loss', figsize=(10, 5))
    plt.show()
    
    # Plot accuracy
    history_data[['accuracy', 'val_accuracy']].plot(title='Accuracy', figsize=(10, 5))
    plt.show()

# Evaluate the model and print classification report
def evaluate_model(model, X_val, Y_val, classes):
    Y_pred = model.predict(X_val)
    
    # Convert predictions and true values from one-hot encoding to class labels
    Y_val_labels = np.argmax(Y_val, axis=1)
    Y_pred_labels = np.argmax(Y_pred, axis=1)
    
    # Print classification report
    print(metrics.classification_report(Y_val_labels, Y_pred_labels, target_names=classes))

    # Display confusion matrix
    confusion_matrix = metrics.confusion_matrix(Y_val_labels, Y_pred_labels)
    print("Confusion Matrix:\n", confusion_matrix)

# Main function to execute the workflow
def main():
    # Load class names (subdirectories)
    classes = os.listdir(DATA_PATH)
    print(f'Classes found: {classes}')

    # Display sample images
    display_sample_images(classes, DATA_PATH)

    # Load and preprocess images
    X, Y = load_images(classes, DATA_PATH, IMG_SIZE)
    
    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=SPLIT_RATIO, random_state=2022)
    print(f'Training data: {X_train.shape}, Validation data: {X_val.shape}')

    # Create the CNN model
    model = create_model(IMG_SIZE)
    model.summary()

    # Train the model
    history = train_model(model, X_train, Y_train, X_val, Y_val, BATCH_SIZE, EPOCHS)

    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    evaluate_model(model, X_val, Y_val, classes)

if __name__ == '__main__':
    main()