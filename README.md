## Project Documentation

### Overview
This project is about an HR assistant that implements question-answering system, on resume of candidates, uploaded via Streamlit-GUI from user, using a retrieval-augmented generation approach. It leverages a conversational model to provide answers based on the context retrieved from documents. The system is designed to be interacted with through a Streamlit web interface, allowing users to upload documents, ask questions, and receive answers.

### System Components

#### 1. Document Loader
- **PDF Loader**: Handles the loading of PDF documents, supporting both single and multiple PDF files. Utilizes libraries such as PyMuPDF for single PDFs and a custom directory loader for multiple PDFs.
- **Web Loader**: Fetches content from web pages based on specified URLs. It uses BeautifulSoup to parse only relevant parts of the web content, focusing on specified HTML classes.

#### 2. Document Processing
- **Chunking**: Documents are divided into manageable chunks using a recursive character text splitter, which can be configured for different chunk sizes and overlaps.
- **Embeddings**: Each chunk is embedded using either OpenAI or HuggingFace embeddings depending on the configuration. This allows for flexibility in choosing the embedding model based on availability or performance criteria.

#### 3. Retrieval System
- **Vector Database**: Stores document embeddings in a FAISS index for efficient similarity search. This setup enables quick retrieval of document chunks that are most relevant to the user's query.
- **Retriever**: Retrieves relevant document chunks based on the query embeddings. It uses a modified marginal ranking (MMR) algorithm to ensure diversity in the retrieved documents.

#### 4. Conversational Model
- **Prompt Settings**: Configures prompts for the conversational model to ensure context-aware interactions. This includes setting up system and human roles in the conversation.
- **Chain Creation**: Combines retrieval and language model chains to form a complete question-answering system. This involves setting up a history-aware retriever and a document-based question answering model.

#### 5. Streamlit Interface
- **Model Selection**: Users can select between different models (e.g., OpenAI models). This choice affects how documents are embedded and how answers are generated.
- **Document Upload**: Users can upload PDF files which are then processed and stored for retrieval. The interface provides feedback on the upload status and any errors.
- **Interaction**: Users can input questions and receive answers generated by the system. The interface supports session management to maintain conversation history and provides interactive elements like checkboxes for file management.


This section of the Streamlit interface is crucial for ensuring a smooth and user-friendly experience while interacting with the question-answering system.


### Streamlit Interface Details

The Streamlit interface is designed to facilitate user interaction with the question-answering system. Below are the key components and functionalities provided by the Streamlit interface as implemented in `app.py`:

#### Model Selection
- Users can select the desired model from a dropdown menu. This selection influences the embedding and answer generation processes.
- If the selected model is from OpenAI, users are prompted to enter their OpenAI API key securely.

#### Document Upload
- Users can upload multiple PDF files simultaneously. Each uploaded file is processed and stored for retrieval.
- The interface provides real-time feedback on the upload status and handles any errors gracefully.

#### Question Input and Response
- Users can type their questions into a chat interface. The system retrieves relevant information and generates responses dynamically.
- The chat interface maintains a session history, allowing users to review past interactions.

#### File Management
- Uploaded files are listed with an option to delete selected files directly from the interface.
- This helps manage the storage and ensures that only relevant documents are kept in the system.

#### System Notifications
- The interface uses notifications and alerts to inform users about the system status, such as model selection confirmation and file processing results.

#### Error Handling
- The system is equipped to handle errors gracefully, providing users with clear error messages and recovery options.


### Usage
To use the system, start the Streamlit application, upload the necessary documents, and input your questions in the chat interface. The system will retrieve relevant information from the uploaded documents and provide answers using the configured conversational model.

Launch Command : `streamlit run app.py`

### Configuration
- **OpenAI API Key**: Required if using OpenAI models for embeddings or language generation.
- **Environment Setup**: Ensure all dependencies are installed, including necessary Python packages and environment variables. The system requires setup of environment variables for API keys and model choices.

### Future Enhancements
- **Semantic Chunking**: Experimental feature to improve the relevance of document chunks by using semantic understanding rather than just text splitting.
- **Enhanced Embeddings**: Explore more advanced embedding techniques for better performance in document retrieval. This could involve using newer models or custom training.

This documentation provides a comprehensive overview of the project's architecture, components, and usage. For detailed configuration and setup instructions, refer to the accompanying setup guides and code comments.

