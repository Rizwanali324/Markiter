import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from huggingface_hub import hf_hub_download

class StreamHandler:
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

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

st.set_page_config(page_title="Your own aiChat!")
st.header("Your own aiChat!")

# Set the system prompt here directly without displaying it
system_prompt = "You are a helpful AI assistant who answers questions in short sentences."

# Create LLM chain to use for our chatbot.
llm_chain = create_chain(system_prompt)

# We store the conversation in the session state.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you today?"}]

# We loop through each message in the session state and render it as a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# We take questions/instructions from the chat input to pass to the LLM
if user_prompt := st.chat_input("Your message here", key="user_input"):
    # Add our input to the session state
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Add our input to the chat window
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Format the prompt using the chain's prompt template
    formatted_prompt = llm_chain.steps[0].format_prompt(question=user_prompt).to_string()

    # Pass our input to the LLM chain and capture the final responses.
    response = llm_chain.invoke({"question": formatted_prompt})

    # Add the response to the session state
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Add the response to the chat window
    with st.chat_message("assistant"):
        st.markdown(response)
