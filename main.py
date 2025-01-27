from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate


pdf_dir = "./pdfs/"

embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model = "deepseek-r1:8b")

def upload_pdf(file):
    pass

def load_pdf(file):
    pass

def split_text(documents):
    pass

def index_docs(documents):
    pass

def retrieve_docs(query):
    pass

def answer_question(question, context):
    pass
