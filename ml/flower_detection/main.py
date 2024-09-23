import numpy as np  
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model

# Parameters
img_size = 224
batch_size = 64
epochs = 30
class_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# Function to create the data generators
def create_data_generators(base_dir, img_size, batch_size):
    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2,  
                                       zoom_range=0.2, horizontal_flip=True, 
                                       validation_split=0.2) 
    val_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2) 
  
    train_data = train_datagen.flow_from_directory(
        base_dir, target_size=(img_size, img_size), 
        subset='training', batch_size=batch_size) 

    val_data = val_datagen.flow_from_directory(
        base_dir, target_size=(img_size, img_size), 
        subset='validation', batch_size=batch_size)

    return train_data, val_data

# Function to build the model
def build_model(input_shape):
    model = Sequential([
        Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dense(5, activation='softmax')  # 5 classes
    ])
    return model

# Function to plot training history
def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    
    plt.savefig("training.png")

# Function to predict the class of an image
def predict_image(model, img_path, class_names):
    img = load_img(img_path, target_size=(224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    
    plt.imshow(img)
    plt.axis('off')
    plt.savefig("prediction_for" + img_path + ".png")

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    
    print(f"Predicted Class: {predicted_class}")

# Main script
base_dir = "data/flowers_data/flowers"
train_data, val_data = create_data_generators(base_dir, img_size, batch_size)

model = build_model(input_shape=(img_size, img_size, 3))
model.summary()

# Plot the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks: Early stopping and saving best model for better performance
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

# Train the model
history = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=callbacks)

# Plot training history
plot_training(history)

# Load the best model
best_model = load_model('best_model.h5')

# Predict the class for test images // img.jpg isn't included in the code, so it can be any image
predict_image(best_model, 'img.jpg', class_names)
