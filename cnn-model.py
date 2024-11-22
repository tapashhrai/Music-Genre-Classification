import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from pyspark.sql import SparkSession
from tensorflow.keras.utils import to_categorical

# Initialize Spark session
spark = SparkSession.builder \
    .appName("CNN Model Training") \
    .config("spark.master", "local") \
    .getOrCreate()

# Paths to the features and labels in HDFS
features_path = "hdfs://namenode:8020/output/features/extracted_features.npy" 
labels_path = "hdfs://namenode:8020/output/features/labels.npy" 

# Load the data from HDFS
def load_data_from_hdfs(path):
    # Use numpy to load the .npy files
    data = np.load(path)
    return data

# Load features and labels
X = load_data_from_hdfs(features_path)
y = load_data_from_hdfs(labels_path)

# Preprocess the data
X = X.astype('float32') / 255.0  # Normalize the images
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)  # Reshape to (num_samples, height, width, channels)

# One-hot encode the labels
y = to_categorical(y, num_classes=10)  # Assuming 10 classes for genres

# Define the CNN model architecture
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=X.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Dropout(0.3))
model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(units=1200, activation='relu'))
model.add(Dropout(0.45))

# Output layer
model.add(Dense(units=10, activation='softmax'))  # 10 classes for genres

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2)

# Save the model to HDFS
h5_model_path = "hdfs://namenode:8020/output/model/Trained_model.h5" 
model.save(h5_model_path)  # Saves the model in .h5 format
print(f"Model saved as H5 format: {h5_model_path}")

# Save the model as Trained_model.keras (Keras' native format)
keras_model_path = "hdfs://namenode:8020/output/model/Trained_model.keras" 
model.save(keras_model_path)  # Saves the model in .keras format
print(f"Model saved as Keras format: {keras_model_path}")

# Stop the Spark session
spark.stop()
