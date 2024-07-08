# PDF-GPT
This repository is an assignment submission and also contains the code for a chatbot using `Langchain` and powered by `OpenAI` using `Streamlit` for frontend UI.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)

## Installation
1. Clone the repository
```
git clone [https://github.com/artzuros/betterzila-assignment.git](https://github.com/artzuros/ChatPDF.git)
```
2. Create a virtual environment
```
python3 -m venv <venv>
```
3. Activate the virtual environment
```
source venv/bin/activate
```
4. Install the requirements
```
pip install -r requirements.txt
```
5. Create a .env file in the root directory
```
OPENAI_API_KEY = <"sk-xxxxx">
```

## User Guide
### Providing documents (PDFs specifically) for the chatbot to refer
- Add your pdfs files to the `dataset` folder.
- Run the following command to start the chatbot.
    ```
    python app.py
    ```
- This will create a `/vectorstore` folder and create new embeddings within the folder.
- Now you can chat with the chatbot with its knowledge of the PDFs provided.
