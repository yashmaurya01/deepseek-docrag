# deepseek-docRAG

The `deepseek-docRAG` project is designed to test the reasoning capabilities of the `deepseek-r1` models for Retrieval-Augmented Generation (RAG) applications. This project leverages the LangChain framework to facilitate document processing and question-answering tasks using PDF documents. The model is run locally using Ollama. More information about the models can be found [here](https://ollama.com/library/deepseek-r1). Please change the `model_name` based on the local compute of your system.

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

- **Imports**: Essential libraries from LangChain and modules for embeddings, language models, and PDF loading.

- **Template**: A prompt template for the question-answering task.

- **PDF Directory**: Specifies where uploaded PDF files are stored.

- **Model Initialization**: Initializes the `deepseek-r1:8b` model for embeddings and language tasks.

- **Functions**:
  - `upload_pdf`: Handles PDF file uploads.
  - `load_pdf`: Loads content from PDF files.
  - `split_text`: Breaks documents into manageable chunks.
  - `index_docs`: Adds documents to the vector store for retrieval.
  - `retrieve_docs`: Performs similarity searches based on user queries.
  - `answer_question`: Constructs prompts and generates answers, extracting content from `<think>` tags.

- **Streamlit Integration**: Provides a file uploader for user interaction.

## Usage
1. **Upload a PDF Document**: Use the provided interface to upload a PDF document.
2. **Ask Questions**: After processing, you can ask questions based on the content of the uploaded document.
3. **Receive Answers**: The model will return concise answers along with any additional reasoning content extracted from the response.

## Conclusion
This project serves as a practical implementation to evaluate the reasoning capabilities of the `deepseek-r1` model in RAG applications, showcasing how to integrate document processing with advanced language models.


