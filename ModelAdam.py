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
    rotation_range=120,
    height_shift_range=0.3,
    zoom_range=0.3,
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True)

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

# Create Adam sequential model
modelAdam = Sequential()
modelAdam.add(Conv2D(filters=32, kernel_size=3, input_shape=(100,100,3)))
modelAdam.add(Activation('relu'))
modelAdam.add(MaxPooling2D(pool_size=(2,2)))
modelAdam.add(Conv2D(filters=64, kernel_size=3))
modelAdam.add(Activation('relu'))
modelAdam.add(Conv2D(filters=128, kernel_size=3))
modelAdam.add(Activation('relu'))
modelAdam.add(MaxPooling2D(pool_size=(2,2)))
modelAdam.add(Conv2D(filters=256, kernel_size=3))
modelAdam.add(Activation('relu'))
modelAdam.add(MaxPooling2D(pool_size=(2,2)))
modelAdam.add(Flatten())
modelAdam.add(Dense(256))
modelAdam.add(Activation('relu'))
modelAdam.add(Dropout(0.5))
modelAdam.add(Dense(128))
modelAdam.add(Activation('relu'))
modelAdam.add(Dropout(0.5))
modelAdam.add(Dense(119))
modelAdam.add(Activation('softmax'))
modelAdam.summary()

# Compile model using adam
modelAdam.compile(optimizer='Adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Define early stopping by tracking the validation loss to avoid overfitting
early_stop = EarlyStopping(monitor="val_loss", patience=3)

# Train model
model_adam = modelAdam.fit(
        training_set, 
        epochs=25,
        validation_data=validation_set,
        callbacks=[early_stop])

# Save model for easy loading 
modelAdam.save('model_ADAM.keras')

# Plot accuracy graph for training and validation
acc_train = model_adam.history['accuracy']
acc_val = model_adam.history['val_accuracy']
plt.plot(acc_train, label = "Train")
plt.plot(acc_val, label = "Validation")
plt.title('Model Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss graph for training and validation
loss_train = model_adam.history['loss']
loss_val = model_adam.history['val_loss']
plt.plot(loss_train, label = "Train")
plt.plot(loss_val, label = "Validation")
plt.title('Model Loss')
plt.legend(loc='upper right')
plt.show()
