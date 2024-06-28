import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Function to set up the model and tokenizer
@st.cache_resource
def setup_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto", 
        trust_remote_code=False,
        revision="main"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model.eval()
    return model, tokenizer

# Function to generate a response from the model
def generate_response(model, tokenizer, prompt, max_new_tokens=140):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=max_new_tokens)
    return tokenizer.batch_decode(outputs)[0]

# Main function for the Streamlit app
def main():
    st.title("Virtual Marketer Assistant")
    
    instructions_string = (
        "virtual marketer assistant, communicates in business, focused on services, "
        "escalating to technical depth upon request. It reacts to feedback aptly and ends responses "
        "with its signature Mr.jon will tailor the length of its responses to match the individual's comment, "
        "providing concise acknowledgments to brief expressions of gratitude or feedback, thus keeping the interaction natural and supportive.\n"
        "Please respond to the following comment."
    )
    
    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    model, tokenizer = setup_model(model_name)
    
    comment = st.text_input("Enter a comment:")
    
    if st.button("Generate Response"):
        if comment:
            prompt_template = lambda comment: f"[INST] {instructions_string} \n{comment} \n[/INST]"
            prompt = prompt_template(comment)
            
            response = generate_response(model, tokenizer, prompt)
            st.write(response)
        else:
            st.warning("Please enter a comment to generate a response.")

if __name__ == "__main__":
    main()
