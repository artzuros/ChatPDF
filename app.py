import streamlit as st
from utils import *
from dotenv import load_dotenv

def main():

    openai_api_key = load_dotenv()
    st.set_page_config(page_title="Chat With files")
    st.header("ChatPDF")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:

        st.session_state.processComplete = None
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.load_local("vectorstore/", embeddings)
        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key) 
        st.session_state.processComplete = True

    if  st.session_state.processComplete == True:
        user_question = st.chat_input("Chat with file 48 LAWS OF POWER")
        if user_question:
            handle_userinput(user_question)


if __name__ == '__main__':
    main()