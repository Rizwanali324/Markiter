import streamlit as st
import os
import tempfile
import soundfile as sf
import speech_recognition as sr
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS

# Function to convert audio to text using speech_recognition
def audio_to_text(audio_data):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_data) as source:
            audio = recognizer.record(source)
            transcript = recognizer.recognize_google(audio, show_all=True)
            if isinstance(transcript, list):
                transcript = transcript[0]['transcript']
        return transcript
    except sr.RequestError:
        return "Error: API unavailable or unresponsive."
    except sr.UnknownValueError:
        return "Error: Unable to recognize speech."

# Function to convert text to speech
def text_to_speech(text, filename="output.mp3"):
    tts = gTTS(text, lang='en')
    tts.save(filename)
    return filename

# Main function for Streamlit interface
def virtual_psychiatrist():
    st.title("Virtual Psychiatrist Assistant")
    st.markdown("Interact with the Virtual Psychiatrist")

    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'ogg', 'flac', 'aac'])

    if uploaded_file is not None:
        try:
            # Create a temporary directory if it doesn't exist
            temp_dir = tempfile.mkdtemp()
            temp_audio = os.path.join(temp_dir, "uploaded_audio.wav")

            # Save the uploaded file as a temporary WAV file
            with open(temp_audio, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Perform speech-to-text conversion
            transcript = audio_to_text(temp_audio)
            
            if "Error" in transcript:
                st.error(transcript)
                return

            # Display the user's spoken text
            st.subheader("Your Comment:")
            st.write(transcript)

            # Optionally, perform further processing or analysis

        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

# Run the Streamlit app
if __name__ == "__main__":
    virtual_psychiatrist()
