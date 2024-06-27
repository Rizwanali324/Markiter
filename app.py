import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import tempfile

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

# Main function for Gradio interface
def virtual_psychiatrist(audio_data):
    # Convert the input audio file to WAV format
    try:
        wav_data = convert_to_wav(audio_data)
    except Exception as e:
        return f"Error converting to WAV: {str(e)}", None

    # Perform speech-to-text conversion
    transcript = audio_to_text(wav_data)
    
    if "Error" in transcript:
        return transcript, None
    
    # Generate response
    prompt = f"""[INST] 
    Please respond to the following comment. 
    {transcript} 
    [/INST]"""

    try:
        # Load tokenizer and model with CPU settings
        tokenizer = AutoTokenizer.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
        model = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", device='cpu')

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        output = model.generate(input_ids=input_ids, max_length=200)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Convert response to speech and provide download link
        audio_file = text_to_speech(response)
        return response, audio_file

    except Exception as e:
        return f"Error generating response: {str(e)}", None

# Helper function to convert audio to WAV using pydub
def convert_to_wav(audio_data):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(audio_data)

    # Create a temporary file to save as WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
        wav_filename = temp_wav_file.name
        audio.export(wav_filename, format="wav")
    
    return wav_filename

# Define Gradio interface
iface = gr.Interface(
    fn=virtual_psychiatrist,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Textbox(label="Generated Response"), gr.Audio(label="Response as WAV")],
    title="Virtual Psychiatrist Assistant",
    description="Interact with the Virtual Psychiatrist"
)

# Launch the Gradio interface
iface.launch()
