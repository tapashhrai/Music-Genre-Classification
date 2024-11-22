import os
import numpy as np
import librosa
from skimage.io import imread
from skimage.transform import resize
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Feature Extraction from Mel Spectrograms") \
    .config("spark.master", "local") \
    .getOrCreate()

# Path to HDFS directory containing Mel spectrogram images
input_dir = "hdfs://namenode:8020/output/melspectrograms/"   # Directory containing Mel spectrogram PNG files
output_dir = "hdfs://namenode:8020/output/features/"   # Output directory for extracted features
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Function to extract features from Mel spectrogram images
def extract_features_from_image(image_path, target_shape=(128, 128)):
    # Load the Mel spectrogram image (PNG)
    spectrogram = imread(image_path)
    
    # Resize the image (if necessary)
    spectrogram_resized = resize(spectrogram, target_shape, mode='reflect', anti_aliasing=True)
    
    # Flatten the image to create a feature vector
    feature_vector = spectrogram_resized.flatten()
    
    # Compute some additional statistical features (mean, variance)
    mean = np.mean(spectrogram_resized)
    variance = np.var(spectrogram_resized)
    
    # Combine the flattened feature vector and statistical features
    features = np.append(feature_vector, [mean, variance])
    
    return features

# Iterate over the Mel spectrogram images and extract features
def save_features_to_hdfs(input_dir, classes, output_dir):
    features_data = []
    labels = []
    
    for i_class, class_name in enumerate(classes):
        class_dir = os.path.join(input_dir, class_name)
        print(f"Processing: {class_name}")

        # Iterate over Mel spectrogram images in the class directory
        for filename in os.listdir(class_dir):
            if filename.endswith('.png'):
                file_path = os.path.join(class_dir, filename)
                
                # Extract features from the Mel spectrogram image
                features = extract_features_from_image(file_path)
                
                # Append features and label
                features_data.append(features)
                labels.append(i_class)

                print(f"Extracted features from {filename}")

    return np.array(features_data), np.array(labels)

# Execute the function to extract features and save them to HDFS
features, labels = save_features_to_hdfs(input_dir, classes, output_dir)

# Save the extracted features and labels to HDFS using PySpark
features_path = "hdfs://namenode:8020/output/features/extracted_features.npy" 
labels_path = "hdfs://namenode:8020/output/features/labels.npy" 

# Create RDDs and save as .npy files (or any other format you need)
rdd_features = spark.sparkContext.parallelize([features])
rdd_labels = spark.sparkContext.parallelize([labels])

rdd_features.saveAsPickleFile(features_path)
rdd_labels.saveAsPickleFile(labels_path)

print("Extracted features and labels have been saved to HDFS.")

# Stop Spark session
spark.stop()
