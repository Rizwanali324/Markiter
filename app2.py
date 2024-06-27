import streamlit as st
from os import path
from pydub import AudioSegment

def mp3_to_wav(src, dst):
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")

def main():
    st.title("MP3 to WAV Converter")

    # File uploader
    st.subheader("Upload your MP3 file:")
    uploaded_file = st.file_uploader("Choose an MP3 file", type=["mp3"])

    if uploaded_file is not None:
        # Save the uploaded file to disk
        with open("temp.mp3", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Convert to WAV
        mp3_to_wav("temp.mp3", "converted.wav")

        # Display download link to converted file
        st.audio("converted.wav", format="audio/wav")

if __name__ == "__main__":
    main()
