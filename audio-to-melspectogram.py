import os
import numpy as np
import librosa
from skimage.transform import resize
from pyspark.sql import SparkSession
from matplotlib import pyplot as plt

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Audio to Mel Spectrogram") \
    .config("spark.master", "local") \
    .getOrCreate()

# Path to HDFS directory containing audio files
data_dir = "hdfs://namenode:8020/input/Data"  # Change this based on your setup
output_dir = "hdfs://namenode:8020/output/melspectrograms/"  # Output directory for spectrogram images
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Function to convert audio files to Mel spectrograms and save as image files
def audio_to_melspectrogram_image(audio_file, output_image_path, target_shape=(128, 128)):
    # Load audio data
    audio_data, sample_rate = librosa.load(audio_file, sr=None)
    
    # Generate Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    
    # Convert amplitude to decibel (log scale)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Resize the Mel spectrogram to the desired shape
    mel_spectrogram_resized = resize(mel_spectrogram_db, target_shape, mode='reflect', anti_aliasing=True)
    
    # Save the Mel spectrogram as a PNG image using matplotlib
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram_resized, aspect='auto', origin='lower', cmap='inferno')
    plt.axis('off')  # Hide axes
    plt.tight_layout()

    # Save the image
    plt.savefig(output_image_path, format='png')
    plt.close()

# Iterate over the audio files and save Mel spectrograms as images
def save_melspectrograms_to_hdfs(data_dir, classes, output_dir):
    for i_class, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        print(f"Processing: {class_name}")

        # Create directory for class if it doesn't exist
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Iterate over audio files in the class directory
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_dir, filename)
                
                # Generate the Mel spectrogram image file path
                spectrogram_filename = f"{filename.split('.')[0]}.png"
                output_image_path = os.path.join(class_output_dir, spectrogram_filename)

                # Convert the audio file to Mel spectrogram and save as image
                audio_to_melspectrogram_image(file_path, output_image_path)

                print(f"Saved Mel spectrogram for {filename} to {output_image_path}")

# Execute the function to save Mel spectrograms as PNG images
save_melspectrograms_to_hdfs(data_dir, classes, output_dir)
