import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
from tensorflow.keras.models import load_model


AUTO = tf.data.experimental.AUTOTUNE

# Function to safely load images
def safe_image_load(image_path):
    try:
        return np.array(Image.open(image_path))
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return np.zeros((224, 224, 3))  # Return a placeholder image

# Load images
images = glob('data/skin_cancer/*/*.jpg')
print(f"Total images: {len(images)}")

# Create data frame [filepath, label]
data = pd.DataFrame({'filepath': images})
data['label'] = data['filepath'].apply(lambda x: x.split('/')[-2])  # Dynamically extract label

# Convert labels to binary
data['label_bin'] = np.where(data['label'].values == 'malignant', 1, 0)

# Make a pie chart
count = data['label'].value_counts()
plt.pie(count.values, labels=count.index, autopct='%1.1f%%')
plt.savefig('pie_chart.png')
plt.show()

# Visualize sample images for each category
for cat in data['label'].unique():
    temp = data[data['label'] == cat]
    index_list = temp.index
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    fig.suptitle(f'Images for {cat} category', fontsize=20)
    
    for i in range(4):
        index = np.random.randint(0, len(index_list))
        index_value = index_list[index]
        
        # Access the data using the correct index
        image_path = data.loc[index_value, 'filepath']
        img = safe_image_load(image_path)  # Load image safely
        ax[i].imshow(img)
        ax[i].set_title(cat)  # Set title for each subplot
        ax[i].axis('off')  # Turn off axes for better visualization

    plt.tight_layout()
    plt.show()

# Separate features from target
features = data['filepath']
target = data['label_bin']

# Split data
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.15, random_state=10)

# Function to decode and preprocess images
def decode_image(filepath, label=None):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0  # Normalize image

    return img, label

# Create TensorFlow datasets
train_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_train, Y_train))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO)
)

val_ds = (
    tf.data.Dataset
    .from_tensor_slices((X_val, Y_val))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
    .prefetch(AUTO)
)

# Load the pre-trained EfficientNetB7 model
pre_trained_model = EfficientNetB7(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False
)

# Freeze pre-trained layers
for layer in pre_trained_model.layers:
    layer.trainable = False

# Define the model architecture
inputs = layers.Input(shape=(224, 224, 3))
x = pre_trained_model(inputs, training=False)  # Use the pre-trained model
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

# Compile the model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['AUC']
)

# Define callbacks for training
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks, verbose=1)

# Save history
with open('history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# Load model
# model = load_model('best_model.h5')

# Convert history to DataFrame for analysis
hist_data = pd.DataFrame(history.history)
print(hist_data.head())

# Plot loss
plt.figure(figsize=(12, 6))
plt.plot(hist_data['loss'], label='Loss')
plt.plot(hist_data['val_loss'], label='Validation Loss')
plt.title('Loss vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('loss_valid_loss.png')
plt.show()


# Plot AUC
plt.figure(figsize=(12, 6))
plt.plot(hist_data['auc'], label='AUC')
plt.plot(hist_data['val_auc'], label='Validation AUC')
plt.title('AUC vs Validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.grid()
plt.savefig('auc_valid_auc.png')
plt.show()

