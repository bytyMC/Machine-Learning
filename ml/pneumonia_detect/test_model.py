from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np

# Load the pre-trained model
model = load_model('our_model.h5')

# Load and preprocess the input image
img_path = '/data/chest_xray/test/PNEUMONIA/person100_bacteria_480.jpeg'
img = image.load_img(img_path, target_size=(224, 224)) 

# Convert the image to a numpy array and expand its dimensions to match the model input
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the image using VGG16's preprocess_input function
img_data = preprocess_input(img_array)

# Make the prediction
prediction = model.predict(img_data)

# Interpret the prediction result
# Assuming binary classification where index 0 is 'safe' and index 1 is 'pneumonia'
if prediction[0][0] > prediction[0][1]:
    print('Prediction: Person is safe.')
else:
    print('Prediction: Person is affected.')

# Display the raw prediction probabilities
print(f'Raw Predictions: {prediction}')