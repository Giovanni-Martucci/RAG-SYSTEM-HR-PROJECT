import bs4
import os
import argparse
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader, PyPDFDirectoryLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_experimental.text_splitter import SemanticChunker 
from langchain_community.vectorstores import FAISS

from utils.constants import Paths, AppModes, PromptForRag



def parse_arguments() -> argparse.Namespace:
    """
    Function to parse the command-line arguments.

    Returns:
        argparse.Namespace: Arguments parsed from the command-line.
    """
    parser = argparse.ArgumentParser(description="Run the RAG chat system.")
    parser.add_argument(
        "--app_mode",
        default=AppModes.STREAMLIT.value,
        help="Choose interface.",
        choices=[AppModes.CLI.value, AppModes.STREAMLIT.value],
    )
    parser.add_argument(
        '-v', 
        '--verbose', 
        action='count', 
        default=0
    )
    return parser.parse_args()

class rag_system:
    def __init__(self, args=0, openai_api_key=None):
        """
        Args:
            args (obj): 
            openai_api_key (str): key mandatory to use llm
        """
        self.args                     = args
        self.openai_api_key           = openai_api_key
        self.path_to_data             = Paths.DOCUMENTS_INPUT_PATH.value
        self.llm                      = self.config_llm_model()
        self.documents                = None
        self.embeddings               = self.config_embeddings()
        self.all_splits               = None
        self.retriever, self.vectordb = None, None
        self.conversational_rag_chain = None
        self.store                    = {}
        
    def setup_documents(self):
        self.documents  = self.load_data()
        self.all_splits = self.doc_chunking()
        self.retriever, self.vectordb = self.doc_embeddings()
        self.conversational_rag_chain = self.prompt_settings()
        
    def print_application_level(self, log_string, err=False, debug=False):
        print('='*50)
        if err:
            print('ERROR:: Application: ' + log_string)
        elif debug:
            print('WARNING:: Application: ' + log_string)
        else:
            print('INFO:: Application: ' + log_string)
        print('='*50)
  
    def config_llm_model(self):
        """
        Function used to read data from source.

        Returns:
            llm (model AI): return a llm model to inference.
        """
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            os.environ["OPENAI_API_MODEL"] = "gpt-4o" #"gpt-3.5-turbo" #"gpt-4-1106-preview"

            # llm OpenAI
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
        else: 
            # llm Llama3
            print("Using Llama3")  
            llm = ChatOllama(model="llama3")

        return llm


    def load_data(self, type="pdf", link=None):
        """
        Function used to read data from source.

        Args:
            type (str, optional): Type of Documents within the Documents directory. Defaults to Paths.INPUT_PATH.value.
            link (str, optional): Input path to the C code. Defaults to Paths.INPUT_PATH.value.

        Returns:
            documents (str): Documents loaded.
        """

        if "pdf" in type:
            try:
                pdf_files = [os.path.join(self.path_to_data, f) for f in os.listdir(self.path_to_data) if f.endswith(".pdf")]
            except:
                self.print_application_level(f"Empty Directory.. please insert documents into correct directory: {self.path_to_data}", err=True)

            if len(pdf_files) > 1:
                # PyPDFLoader for multiple pdfs
                loader = PyPDFDirectoryLoader(Paths.DOCUMENTS_INPUT_PATH.value)                
            else:  
                # PyMuPdrLoader used for single pdf file
                loader = PyMuPDFLoader(pdf_files[0])   
        elif "web" in type:
            loader = WebBaseLoader(web_paths=(link,),bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),)
        
        documents = loader.load()

        return documents

    def doc_chunking(self):
        """
        This function divide documents into chunk (pieces of document).

        Args:
            documents (str): Input data from user.
        
        Returns:
            all_splits (list): List of chunks.
        """

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,) 
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300,) 
        all_splits = text_splitter.split_documents(self.documents)

        # Semantic chunking (experimental)
        # semantic_chunker = SemanticChunker(self.embeddings, breakpoint_threshold_type="percentile")
        # semantic_chunks = semantic_chunker.create_documents([d.page_content for d in self.documents])
        return all_splits #semantic_chunks
    
    def config_embeddings(self):
        """
        Function used to read data from source.

        Returns:
            llm (model AI): return a llm model to inference.
        """

        if self.openai_api_key:
            embeddings               = OpenAIEmbeddings(model="text-embedding-3-large")
        else:
            model_name               = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            model_kwargs             = {"device": "cuda"} #cuda for gpu | cpu for c
            embeddings               = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        
        return embeddings
         

    def doc_embeddings(self):
        """
        This function apply the embedding algorithm to the documents (chunked).
        
        Returns:
            retrivier (obj): 
            vectordb   (db):
        """
        
        # Load vectors_embedding into Vectordb  
        vectordb = FAISS.from_documents(self.all_splits, self.embeddings)
        
        # With persistence:
        # vectordb.save_local("faiss_persistence")
        # vectordb = FAISS.load_local("faiss_persistence", self.embeddings)        

        # Load retrivier from vectordb
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 6})

        return retriever, vectordb

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Use this function to handle the sessions of users

        Args:
            session_id (str): unique session_id for each user
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def prompt_settings(self):
        """
        Function to load prompts for memory and docs. 
        Generate conversational_rag_chain object to invoke the chain created. 

        """
        # Memory usage settings
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PromptForRag.CONTEXTUALIZE_Q_SYSTEM_PROMPT.value),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever( # Domanda contestualizzata dalla chat history, utilizzata poi dal retriever per estrarre i documenti corretti
            self.llm, self.retriever, contextualize_q_prompt
        )


        # Documents usage settings
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PromptForRag.qa_system_prompt_cv.value),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )


        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        return conversational_rag_chain

    def ask(self, user_request, session_id="abc123"):
        """
        This function permit to ask the question to the whole system. 
        This is the method used from each other external system.

        Args:
            user_request (str): this is the quesiton of user
            session_id   (str): unique session_id for each user
        
        Returns:
            answer (str): answer generated using Documents elements
        """
       
        return self.conversational_rag_chain.invoke(
                {"input": user_request},
                config={"configurable": {"session_id": session_id}}, #client.session_id cercare un id univoco per ogni sessione
        )["answer"]

    def run_cli(self):
        """
        Run the chatbot
        """
        while True:
            user_request = input("[*] - Inserisci la tua richiesta: ")            
            
            answer = self.ask(user_request)
            print(f"\n\n[+] - REQUEST:\n{user_request}\n[*] - ANSWER:\n{answer}\n")



#############################

if __name__ == "__main__":

    args = parse_arguments()
    rag = rag_system(args=args)
    rag.run_cli()


    


    


    


    



