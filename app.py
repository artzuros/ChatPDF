import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.callbacks import get_openai_callback


def main():

    openai_api_key = st.secrets['OPENAI_API_KEY']
    st.set_page_config(page_title="Chat With files")
    st.header("Betterzila Assignment developed by :blue[PRANAV BANSAL]")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

        path = 'dataset/48lawsofpower.pdf'
        files_text = get_pdf_text(path)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key) 

        st.session_state.processComplete = True

    if  st.session_state.processComplete == True:
        user_question = st.chat_input("Chat with file 48 LAWS OF POWER")
        if user_question:
            handel_userinput(user_question)

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_csv_text(file):
    return "a"

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks,embeddings)
    return knowledge_base

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# This function takes a user question as input, sends it to a conversation model and displays the conversation history along with some additional information.
def handel_userinput(user_question):

    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))
    #     st.write(f"Total Tokens: {cb.total_tokens}" f", Prompt Tokens: {cb.prompt_tokens}" f", Completion Tokens: {cb.completion_tokens}" f", Total Cost (USD): ${cb.total_cost}")




if __name__ == '__main__':
    main()
