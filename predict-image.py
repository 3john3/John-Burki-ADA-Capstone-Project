# Import libraries for image prediction
import os
import numpy as np
import keras
from keras.models import load_model
import tensorflow as tf
from keras.utils import img_to_array, load_img

# import path to foulders 
categories_course = '/Users/johnstefanoburki/Desktop/UNIL/Classes/Semester 2/Data-Science/Final-Project/Code/Database/Validation'
test_image_path = '/Users/johnstefanoburki/Desktop/UNIL/Classes/Semester 2/Data-Science/Final-Project/Code/Database/Test/45.jpg'

# import categories 
categories = os.listdir(categories_course)
if '.DS_Store' in categories:
  categories.remove('.DS_Store')

# import all three models 
model_saved_SGD = keras.models.load_model('model_SGD_32.keras')
model_saved_Adam = keras.models.load_model('model_ADAM.keras')
model_saved_RMSprop = keras.models.load_model('model_RMSprop.keras')

# Import testing image
new_image = keras.preprocessing.image.load_img(test_image_path, target_size=(100,100))
new_image_array = keras.preprocessing.image.img_to_array(new_image)
new_image_array = np.array([new_image_array])
new_image.show()

# Predict using final model with SGD optimizer 
prediction_SGD = model_saved_SGD.predict(new_image_array)
category_SGD = np.argmax(prediction_SGD, axis =1)
output_SGD = categories[category_SGD[0]]
print("The SGD model predicts the image being tested is a: "+ output_SGD)

# Predict using final model with Adam optimizer 
prediction_Adam = model_saved_Adam.predict(new_image_array)
category_Adam = np.argmax(prediction_Adam, axis =1)
output_Adam = categories[category_Adam[0]]
print("The Adam model predicts the image being tested is a: "+ output_Adam)

# Predict using final model with RMSprop optimizer 
prediction_RMS = model_saved_RMSprop.predict(new_image_array)
category_RMS = np.argmax(prediction_RMS, axis =1)
output_RMS = categories[category_RMS[0]]
print("The RMS model predicts the image being tested is a: "+ output_RMS)
