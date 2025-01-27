from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import re

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer: 
"""

pdf_dir = "./pdfs/"

model_name = "deepseek-r1:8b"

embeddings = OllamaEmbeddings(model=model_name)
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model = model_name)

def upload_pdf(file):
    with open(pdf_dir + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True,
    )

    return text_splitter.split_documents(documents)

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    ans = chain.invoke({"question": question, "context": context})

    # Extract text between <think> and </think> tags
    think_text = re.search(r'<think>(.*?)</think>', ans, re.DOTALL)
    if think_text:
        think_content = think_text.group(1)
        final_answer = ans.replace(think_content, '').strip()  # Get the final answer
    else:
        think_content = None  # or handle the case where no <think> tags are found
        final_answer = ans  # If no think content, return the original answer

    return final_answer, think_content  # Return final answer and extracted think content


uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False,
)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdf_dir + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)
    st.chat_message("assistant").write("Ask me a complex question about the uploaded document")
    question = st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        final_ans, think_content = answer_question(question, related_documents)
        
        # Display internal thought with an avatar
        st.chat_message("internal thought", avatar="internal.png").write(think_content)
        
        # Display only the final answer
        st.chat_message("assistant").write(final_ans)

