import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("vectorstore")

def get_conversational_chain():
    prompt_template = """ You are a helpful chatbot, talk to the user and
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context" , don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

chat_history = []

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("vectorstore", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    chat_history.append({"role": "user", "content": user_question})
    chat_history.append({"role": "bot", "content": response['output_text']})

    print(response)

st.title('Betterzila Assignment by :blue[Pranav Bansal]')

dataset_folder = "dataset/"
if os.path.exists(dataset_folder):
    for filename in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, filename)
        # Process each file in the folder
        file_details = {"FileName": filename, "FilePath": file_path}
        st.write(file_details)
        raw_text = get_pdf_text([file_path])
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

user_question = st.text_input("Enter your question here")
if user_question:
    user_input(user_question)
else:
    st.warning('Please enter a question.')

for message in chat_history:
    if message["role"] == "user":
        st.markdown((":red[USER]"))
        st.markdown((message["content"]))
    else:
        st.markdown((":green[BOT]"))
        st.markdown((message["content"]))