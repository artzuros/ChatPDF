from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message
from langchain.callbacks import get_openai_callback

# Create Vector Store and Index
def embed_all(DATASET = "dataset/", FAISS_INDEX = "vectorstore/"):
    """
    Embed all files in the dataset directory
    """
    # Create the document loader
    loader = DirectoryLoader(DATASET, glob="*.pdf", loader_cls=PyPDFLoader)
    # Load the documents
    documents = loader.load()
    # Create the splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    # Split the documents into chunks
    chunks = splitter.split_documents(documents)
    # Load the embeddings
    embeddings = HuggingFaceEmbeddings()
    # Create the vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    # Save the vector store
    vector_store.save_local(FAISS_INDEX)

def get_conversation_chain(vectorstore,openai_api_key):

    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# This function takes a user question as input, sends it to a conversation model and displays the conversation history along with some additional information.
def handle_userinput(user_question):

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