import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Retrieve the API key from Streamlit secrets
api_key = st.secrets['secrets']["API_KEY"]

def get_llm_response(query, chat_history):
    template = """
    You are a supportive psychologist assistant. Engage in a warm and empathetic conversation, offering thoughtful advice and strategies for the user's mental and emotional well-being. Reflect on the history of our conversation to provide personalized guidance.
    
    Chat history: {chat_history}
    
    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=api_key,max_tokens=500)
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_question": query
    })

def main():
    st.set_page_config(page_title='Psychologist', page_icon='pyscho.jpg')
    st.header(" Your supportive psychologist assistant",)
    st.sidebar.markdown("# Aibytec")
    
    st.sidebar.image('logo.jpg', width=200)
    # Initialize chat history if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content='Hello, I am your supportive psychologist assistant. How can I assist you today?')
        ]
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = []

    # Sidebar with "New Chat" button and chat history list
    with st.sidebar:
        st.title("Options")

        # New Chat button
        if st.button("New Chat"):
            st.session_state.chat_histories.append(st.session_state.chat_history)
            st.session_state.chat_history = [
                AIMessage(content='Hello, I am your supportive psychologist assistant. How can I assist you today?')
            ]

        # Display previous chats
        if st.session_state.chat_histories:
            st.subheader("Chat History")
            for i, hist in enumerate(st.session_state.chat_histories):
                if st.button(f"Chat {i + 1}", key=f"chat_{i}"):
                    st.session_state.chat_history = hist

    # Display the conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('Human'):
                st.write(message.content)

    user_input = st.chat_input('Type your question or request here...')

    if user_input is not None and user_input != "":
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        with st.chat_message("Human"):
            st.markdown(user_input)

        with st.chat_message("AI"):
            # Generate and display the response
            response_gen = get_llm_response(user_input, st.session_state.chat_history)
            response = ''.join(list(response_gen))
            st.write(response)

        st.session_state.chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()
