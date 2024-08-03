import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from huggingface_hub import hf_hub_download
from datetime import datetime
import json
import os

# Load and save chat history to a file
CHAT_HISTORY_FILE = "chat_history.json"

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    return []

def save_chat_history(messages):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(messages, file)

# Define the StreamHandler to intercept streaming output from the LLM.
class StreamHandler:
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Cache the LLM chain creation
@st.cache_resource
def create_chain(system_prompt):
    repo_id, model_file_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF", "mistral-7b-instruct-v0.1.Q4_0.gguf"
    model_path = hf_hub_download(repo_id=repo_id, filename=model_file_name, repo_type="model")

    llm = LlamaCpp(
        model_path=model_path,
        temperature=0,
        max_tokens=512,
        top_p=1,
        stop=["[INST]"],
        verbose=False,
        streaming=True,
    )

    template = f"<s>[INST]{system_prompt}[/INST]</s>\n\n[INST]{{question}}[/INST]"
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = prompt | llm

    return llm_chain

# Set the webpage title and header
st.set_page_config(page_title="Your own aiChat!")
st.header("Your own aiChat!")

# Define the system prompt without displaying it
system_prompt = "You are a helpful AI assistant who answers questions in short sentences."

# Create LLM chain to use for our chatbot.
llm_chain = create_chain(system_prompt)

# Initialize or load the conversation history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()
    if not st.session_state.messages:
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you today?"}]

# Function to display chat messages
def display_chat_messages():
    for message in st.session_state.messages:
        # Add a timestamp if it doesn't exist
        timestamp = message.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        with st.chat_message(message["role"]):
            st.markdown(f"**{message['role']}** ({timestamp}):\n\n{message['content']}")

# Function to clear the chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. How may I help you today?", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]
    save_chat_history(st.session_state.messages)
    st.experimental_rerun()

# Display the chat messages
display_chat_messages()

# Input box for user messages
if user_prompt := st.chat_input("Your message here", key="user_input"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append({"role": "user", "content": user_prompt, "timestamp": timestamp})

    with st.chat_message("user"):
        st.markdown(f"**user** ({timestamp}):\n\n{user_prompt}")

    # Format the prompt and generate a response
    formatted_prompt = llm_chain.steps[0].format_prompt(question=user_prompt).to_string()
    response = llm_chain.invoke(formatted_prompt)  # Pass the string directly

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": timestamp})

    with st.chat_message("assistant"):
        st.markdown(f"**assistant** ({timestamp}):\n\n{response}")

    # Save the updated chat history
    save_chat_history(st.session_state.messages)

# Add a button to clear the chat history
if st.button("Clear Chat History"):
    clear_chat_history()
