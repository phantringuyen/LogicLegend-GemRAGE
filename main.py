import os, tempfile
from pathlib import Path

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.globals import set_debug
set_debug(True)

import google.generativeai as genai
import requests
import streamlit as st
from keys import *

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("data", "vector_store")
)

st.set_page_config(page_title="RAG")
st.title("GemRAGE - Retrieval Augmented Generation Engine")


def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob="**/*.pdf")
    documents = loader.load()
    return documents


def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    '''
    This function creates embeddings for the documents and stores them in a local vector store.
    '''
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(texts)
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

def query_llm(model, retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
    )
    result = qa_chain({"question": query, "chat_history": st.session_state.messages})
    result = result["answer"]
    st.session_state.messages.append((query, result))
    return result


def input_fields():

    st.session_state.source_docs = st.file_uploader(
        label="Upload Documents", type="pdf", accept_multiple_files=True
    )



def process_documents():
    if not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            for source_doc in st.session_state.source_docs:

                with tempfile.NamedTemporaryFile(
                    delete=False, dir=TMP_DIR.as_posix(), suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(source_doc.read())

                documents = load_documents()

                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()

                texts = split_documents(documents)

                st.session_state.retriever = embeddings_on_local_vectordb(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")


def get_conversational_chain():
    # Define a prompt template for asking questions based on a given context
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    # Initialize a ChatGoogleGenerativeAI model for conversational AI
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # Create a prompt template with input variables "context" and "question"
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Load a question-answering chain with the specified model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    # Create embeddings for the user question using a Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load a FAISS vector database from a local file
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search in the vector database based on the user question
    docs = new_db.similarity_search(user_question)

    # Obtain a conversational question-answering chain
    chain = get_conversational_chain()

    # Use the conversational chain to get a response based on the user question and retrieved documents
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    # Display the response in a Streamlit app (assuming 'st' is a Streamlit module)
    # st.write("Reply: ", response["output_text"])
    if check_QA(user_question, response['output_text']) == 'safe':
        st.session_state.messages.append((user_question, response['output_text']))
    else:
        st.session_state.messages.append((user_question, "**I'm sorry, I can't answer that question.**"))

    return response['output_text']

def check_QA(question, answer):
    dictToSend = {'Text': question, 'Answer': answer}
    res = requests.post(URL, json=dictToSend)
    print(res.text)
    return res.text

def boot():
    genai.configure(api_key=GOOGLE_API_KEY)

    input_fields()

    st.button("Submit Documents", on_click=process_documents)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message("human").write(message[0])
        st.chat_message("ai").write(message[1])

    if query := st.chat_input():
        st.chat_message("human").write(query)
        # response = query_llm(model, st.session_state.retriever, query)
        response = user_input(query)
        st.chat_message("ai").write(response)


if __name__ == "__main__":
    boot()
