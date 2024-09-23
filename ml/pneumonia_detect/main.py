from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image shape and VGG16 base model
IMAGESHAPE = [224, 224, 3]  # Image size 224x224 with 3 channels for RGB
vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False)

# Set all layers in VGG16 to non-trainable
for each_layer in vgg_model.layers:
    each_layer.trainable = False

# Define paths for training and testing data
training_data = 'data/chest_xray/train'
testing_data = 'data/chest_xray/test'

# Get the number of classes (folders in train directory)
classes = glob(f'{training_data}/*')

# Add flattening and prediction layers
flatten_layer = Flatten()(vgg_model.output)
prediction = Dense(len(classes), activation='softmax')(flatten_layer)

# Create final model
final_model = Model(inputs=vgg_model.input, outputs=prediction)
final_model.summary()

# Compile the model with Adam optimizer and categorical crossentropy loss
final_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Prepare the data generators for training and testing datasets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from the directories
training_set = train_datagen.flow_from_directory(
    training_data,
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    testing_data,
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical'
)

# Fit the model
fitted_model = final_model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Plot training and validation loss
plt.plot(fitted_model.history['loss'], label='Training Loss')
plt.plot(fitted_model.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('Loss_Val_loss.png')
plt.show()

# Plot training and validation accuracy
plt.plot(fitted_model.history['accuracy'], label='Training Accuracy')
plt.plot(fitted_model.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('Acc_Val_acc.png')
plt.show()

# Save the model
final_model.save('our_model.h5')