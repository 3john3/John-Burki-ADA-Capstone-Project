# Import necessary libraries
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.layers import Activation
from keras.callbacks import EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Load directories for training and valididation
train_path = '/Users/johnstefanoburki/Desktop/UNIL/Classes/Semester 2/Data-Science/Final-Project/Code/Database/Training'
valid_path = '/Users/johnstefanoburki/Desktop/UNIL/Classes/Semester 2/Data-Science/Final-Project/Code/Database/Validation'

# Augment training images using various alterations
train_augmentation = ImageDataGenerator(
    rotation_range=110,
    zoom_range=0.3,
    rescale=1./255,
    horizontal_flip=True)

valid_augmentation = ImageDataGenerator(rescale=1./255)

# Split data into training set 
training_set = train_augmentation.flow_from_directory(
    train_path,
    shuffle=True,
    batch_size=32,
    target_size=(100, 100),
    color_mode='rgb',
    class_mode="categorical")

# Split data into validation set 
validation_set = valid_augmentation.flow_from_directory(
    valid_path,
    batch_size=32,
    target_size=(100, 100),
    color_mode='rgb',
    class_mode="categorical")

# Create variant of second sequential model
modeltrain2b = Sequential()
modeltrain2b.add(Conv2D(filters=32, kernel_size=3, input_shape=(100,100,3)))
modeltrain2b.add(Activation('relu'))
modeltrain2b.add(MaxPooling2D(pool_size=(2,2)))
modeltrain2b.add(Conv2D(filters=64, kernel_size=3))
modeltrain2b.add(Activation('relu'))
modeltrain2b.add(MaxPooling2D(pool_size=(2,2)))
modeltrain2b.add(Conv2D(filters=128, kernel_size=3))
modeltrain2b.add(Activation('relu'))
modeltrain2b.add(MaxPooling2D(pool_size=(2,2)))
modeltrain2b.add(Flatten())
modeltrain2b.add(Dense(128))
modeltrain2b.add(Activation('relu'))
modeltrain2b.add(Dropout(0.5))
modeltrain2b.add(Dense(119))
modeltrain2b.add(Activation('softmax'))

# Summary of the model
modeltrain2b.summary()

# Compile model using SGD as comparison standard
modeltrain2b.compile(optimizer='SGD',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Define early stopping by tracking the validation loss to avoid overfitting
early_stop = EarlyStopping(monitor="val_loss", patience=3)

# Train model
model_trial_2b_train = modeltrain2b.fit(
        training_set, 
        epochs=25,
        validation_data=validation_set,
        callbacks=[early_stop])

# Plot accuracy graph for training and validation
acc_train = model_trial_2b_train.history['accuracy']
acc_val = model_trial_2b_train.history['val_accuracy']
plt.plot(acc_train, label = "Train")
plt.plot(acc_val, label = "Validation")
plt.title('Model Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss graph for training and validation
loss_train = model_trial_2b_train.history['loss']
loss_val = model_trial_2b_train.history['val_loss']
plt.plot(loss_train, label = "Train")
plt.plot(loss_val, label = "Validation")
plt.title('Model Loss')
plt.legend(loc='upper right')
plt.show()