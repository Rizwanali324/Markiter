import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from gtts import gTTS
import os
import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the model and tokenizer
@st.cache(allow_output_mutation=True)
def setup_model(model_name):
    logging.info('Setting up model and tokenizer.')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=False,
        revision="main"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model.eval()
    logging.info('Model and tokenizer setup completed.')
    return model, tokenizer

# Generate a response from the model
def generate_response(model, tokenizer, prompt, max_new_tokens=140):
    logging.info('Generating response for the prompt.')
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=max_new_tokens)
    response = tokenizer.batch_decode(outputs)[0]
  
    # Extract only the response part (assuming everything after the last newline belongs to the response)
    response_parts = response.split("\n")
    logging.info('Response generated.')
    return response_parts[-1]  # Return the last element (response)

# Remove various tags using regular expressions
def remove_tags(text):
    logging.info('Removing tags from the text.')
    tag_regex = r"<[^>]*>"  # Standard HTML tags
    custom_tag_regex = r"<.*?>|\[.*?\]|{\s*?\(.*?\)\s*}"  # Custom, non-standard tags (may need adjustments)
    all_tags_regex = f"{tag_regex}|{custom_tag_regex}"  # Combine patterns
    cleaned_text = re.sub(all_tags_regex, "", text)
    logging.info('Tags removed.')
    return cleaned_text

# Generate the audio file
def text_to_speech(text, filename="response.mp3"):
    logging.info('Generating speech audio file.')
    tts = gTTS(text)
    tts.save(filename)
    logging.info('Speech audio file saved.')
    return filename

# Main function for the Streamlit app
def main():
    st.title("Virtual Marketer Assistant")
    st.write("Enter a comment and get a response from the virtual marketer assistant. Download the response as an MP3 file.")

    comment = st.text_area("Enter a comment", height=100)

    if st.button("Generate Response"):
        logging.info('Main function triggered.')
        instructions_string = (
            "virtual marketer assistant, communicates in business, focused on services, "
            "escalating to technical depth upon request. It reacts to feedback aptly and ends responses "
            "with its signature Mr.jon will tailor the length of its responses to match the individual's comment, "
            "providing concise acknowledgments to brief expressions of gratitude or feedback, thus keeping the interaction natural and supportive.\n"
        )

        model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
        try:
            model, tokenizer = setup_model(model_name)

            if comment:
                prompt_template = lambda comment: f"[INST] {instructions_string} \n{comment} \n[/INST]"
                prompt = prompt_template(comment)

                response = generate_response(model, tokenizer, prompt)
                response_without_tags = remove_tags(response)
                response_without_inst = response_without_tags.rstrip("[/INST]")

                audio_file = text_to_speech(response_without_inst)
                
                st.write(response_without_inst)
                st.audio(audio_file)
            else:
                st.warning('Please enter a comment to generate a response.')
        except Exception as e:
            logging.error(f'Error occurred: {str(e)}')
            st.error("An error occurred. Please try again later.")

if __name__ == "__main__":
    main()
