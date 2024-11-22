import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import tempfile
import matplotlib.pyplot as plt
from tensorflow.image import resize
import librosa.display

# Function to load the model
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model("Trained_model.keras")
    return model

# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data), audio_data, sample_rate

# TensorFlow Model Prediction
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Prediction"])

## Main Page
if app_mode == "Home":
    st.markdown("""<style>.stApp { background-color: #181646; color: white; } h2, h3 { color: white; }</style>""", unsafe_allow_html=True)

    st.markdown(''' ## Welcome to the,\n ## Music Genre Classification System! 🎶🎧''')
    image_path = "music_genre_home.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""**Our goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and our system will analyze it to detect its genre. Discover the power of AI in music analysis!**""")

elif app_mode == "Prediction":
    st.header("Model Prediction")
    
    # Allow both .mp3 and .wav files
    test_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if test_audio is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(test_audio.read())
            filepath = tmpfile.name

        # Check if 'play_audio' state is in session_state
        if 'play_audio' not in st.session_state:
            st.session_state['play_audio'] = False
        
        # Check if 'prediction' state is in session_state
        if 'prediction' not in st.session_state:
            st.session_state['prediction'] = None
        
        # Show Button to play audio
        if st.button("Play Audio"):
            st.session_state['play_audio'] = True
            st.session_state['prediction'] = None  # Reset prediction when playing new audio

        if st.session_state['play_audio']:
            st.audio(filepath)

        # Predict Button
        if st.button("Predict"):
            with st.spinner("Please Wait.."):
                # Load and preprocess the uploaded audio file
                X_test, audio_data, sample_rate = load_and_preprocess_data(filepath)
                
                # Get model prediction
                result_index = model_prediction(X_test)
                st.balloons()

                # Define possible genres
                label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
                genre = label[result_index]
                
                # Store prediction result in session_state
                st.session_state['prediction'] = genre

                # Display Mel Spectrogram
                st.subheader("Mel Spectrogram of the Uploaded Audio")
                mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
                log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

                plt.figure(figsize=(10, 4))
                librosa.display.specshow(log_mel_spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Mel Spectrogram')
                st.pyplot(plt)

                # Display Waveform of the Uploaded Audio
                st.subheader("Waveform of the Uploaded Audio")
                plt.figure(figsize=(10, 4))
                librosa.display.waveshow(audio_data, sr=sample_rate, alpha=0.5)
                plt.title('Audio Waveform')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                st.pyplot(plt)

                # Display Chroma Feature
                st.subheader("Chroma Feature of the Uploaded Audio")
                chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
                plt.colorbar()
                plt.title('Chroma Feature')
                st.pyplot(plt)

                # Display Tonnetz (Harmonic Relations) Feature
                st.subheader("Tonnetz (Harmonic Relations) of the Uploaded Audio")
                tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sample_rate)
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(tonnetz, y_axis='tonnetz', x_axis='time')
                plt.colorbar()
                plt.title('Tonnetz (Harmonic Relations)')
                st.pyplot(plt)

        # Display Prediction Result
        if st.session_state['prediction']:
            # Define background color for black
            black_bg = "#000000"
            # Apply black background with white text
            st.markdown(f"""
            <div style="background-color:{black_bg}; padding: 30px; border-radius: 15px; text-align: center;">
                <h2 style="color:white;">🎶 **Model Prediction**</h2>
                <h3 style="color:white; font-weight:bold;">It's a <span style="color:white">{st.session_state['prediction'].capitalize()}</span> music!</h3>
                <h5 style="color:white;">Enjoy the rhythm of {st.session_state['prediction'].capitalize()} genre! 🎧</h5>
            </div>
            """, unsafe_allow_html=True)
