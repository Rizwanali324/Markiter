import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import os

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

# Function to convert text to speech using gTTS and save as WAV
def text_to_speech(text, filename="output.wav"):
    tts = gTTS(text, lang='en')
    tts.save(filename)
    return filename

# Main function for Streamlit app
def main():
    st.title("Virtual Psychiatrist Assistant")
    st.markdown("Interact with the Virtual Psychiatrist")

    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])

    if uploaded_file is not None:
        try:
            # Create temporary directory if it doesn't exist
            temp_dir = "temp"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # Save the uploaded file locally
            audio_data = os.path.join(temp_dir, uploaded_file.name)
            with open(audio_data, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Convert uploaded audio to WAV using pydub if it's not already in WAV format
            if uploaded_file.type != "audio/wav":
                audio_data = convert_to_wav(audio_data)

            # Perform speech-to-text conversion
            transcript = audio_to_text(audio_data)

            if "Error" in transcript:
                st.error(transcript)
                return

            # Display the user's spoken text
            st.subheader("Your Comment:")
            st.write(transcript)

            # Generate response
            prompt = f"""[INST] 
            Please respond to the following comment. 
            {transcript} 
            [/INST]"""

            # Load tokenizer and model with CPU settings
            tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
            model = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", device='cpu')

            input_ids = tokenizer(prompt, return_tensors='pt').input_ids
            output = model.generate(input_ids=input_ids, max_length=200)
            response = tokenizer.decode(output[0], skip_special_tokens=True)

            # Display generated response
            st.subheader("Generated Response:")
            st.write(response)

            # Convert response to speech and provide download link
            audio_file = text_to_speech(response)
            st.audio(audio_file, format='audio/wav')

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Helper function to convert audio to WAV using pydub
def convert_to_wav(audio_data):
    audio = AudioSegment.from_file(audio_data)

    # Create a temporary file to save as WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
        wav_filename = temp_wav_file.name
        audio.export(wav_filename, format="wav")

    return wav_filename

if __name__ == "__main__":
    main()
