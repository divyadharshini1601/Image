import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Load and preprocess the input image
img_path = 'path/to/your/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Perform the prediction
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]

# Print the predictions
for _, label, confidence in decoded_preds:
    print(f"{label}: {confidence * 100}%")
