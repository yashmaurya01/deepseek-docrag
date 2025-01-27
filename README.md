# deepseek-docRAG

The `deepseek-docRAG` project is designed to test the reasoning capabilities of the `deepseek-r1` models for Retrieval-Augmented Generation (RAG) applications. This project leverages the LangChain framework to facilitate document processing and question-answering tasks using PDF documents.

## Installation

To get started, follow these steps:

1. **Clone the Repository**: 
   Clone this repository to your local machine.

2. **Install Requirements**: 
   Navigate to the project directory and run the following command to install the necessary packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Pull the Model**: 
   Use Ollama to pull the required model:

   ```bash
   ollama pull deepseek-r1:8b
   ```

4. **Run the Application**: 
   Start the Streamlit application with the following command:

   ```bash
   streamlit run main.py
   ```

## Code Explanation

### main.py

- **Imports**: The script begins by importing necessary libraries from LangChain and other modules, including `OllamaEmbeddings`, `OllamaLLM`, and `PDFPlumberLoader`, among others.

- **Template Definition**: A prompt template is defined for the question-answering task. It instructs the model on how to respond based on the retrieved context.

- **PDF Directory**: The directory for storing uploaded PDF files is specified as `pdf_dir`.

- **Model Initialization**: The `deepseek-r1:8b` model is initialized for both embeddings and language model tasks.

- **PDF Upload Function**: The `upload_pdf` function handles the uploading of PDF files, saving them to the specified directory.

- **PDF Loading Function**: The `load_pdf` function uses `PDFPlumberLoader` to load the content of the uploaded PDF files into a format suitable for processing.

- **Text Splitting Function**: The `split_text` function utilizes `RecursiveCharacterTextSplitter` to break down the loaded documents into manageable chunks for better processing.

- **Indexing Documents**: The `index_docs` function adds the split documents to the in-memory vector store for efficient retrieval.

- **Document Retrieval**: The `retrieve_docs` function performs a similarity search in the vector store based on the user's query.

- **Answering Questions**: The `answer_question` function constructs a prompt using the retrieved context and invokes the model to generate an answer. It also extracts any content enclosed within `<think>` tags from the model's response.

- **Streamlit File Uploader**: The application provides a file uploader for users to upload PDF documents, which will then be processed and used for question-answering.

## Usage
1. **Upload a PDF Document**: Use the provided interface to upload a PDF document.
2. **Ask Questions**: After processing, you can ask questions based on the content of the uploaded document.
3. **Receive Answers**: The model will return concise answers along with any additional reasoning content extracted from the response.

## Conclusion
This project serves as a practical implementation to evaluate the reasoning capabilities of the `deepseek-r1` model in RAG applications, showcasing how to integrate document processing with advanced language models.


