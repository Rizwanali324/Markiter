import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from transformers import AutoModelForCausalLM, AutoTokenizer
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
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

# Function to convert mp3 to wav using pydub
def convert_mp3_to_wav(mp3_file):
    try:
        audio = AudioSegment.from_file(mp3_file)
        wav_filename = os.path.splitext(mp3_file)[0] + ".wav"
        audio.export(wav_filename, format="wav")
        return wav_filename
    except Exception as e:
        return str(e)

# Function to convert text to speech using gTTS
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
            # Create temporary file to store audio
            temp_audio = NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio.close()

            # Save uploaded file locally
            with open(temp_audio.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Check file type and convert if necessary
            if uploaded_file.type == 'audio/mp3':
                converted_file = convert_mp3_to_wav(temp_audio.name)
                if converted_file is None:
                    st.error("Error converting MP3 to WAV.")
                    return
                audio_data = converted_file
            else:
                audio_data = temp_audio.name

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

if __name__ == "__main__":
    main()
