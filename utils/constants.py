"""
This module contains the Enum classes for the MetaGPT model.
"""

from enum import Enum
import os

class Paths(Enum):
    """
    Enum class for the paths used in the RAG system.
    """

    FOLDER_PATH           = os.getcwd()
    DOCUMENTS_INPUT_PATH  = f"{FOLDER_PATH}/documents"


class AppModes(Enum):
    """
    Enum class for the modes of the MetaGPT model.
    """

    CLI = "cli"
    STREAMLIT = "streamlit"


class PromptForRag(Enum):
    """
    Enum class for the actions of the SimpleWriteDocument actor.
    """


    ### Contextualize input question with chat_history ###
    CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""



    qa_system_prompt_cv = """You are an Human Resources assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. Answer in the same language of the quesiton. \
    
    Question: {input} 
    Context: {context} 
    Answer:"""



# Streamlit Constants
class StreamlitConstants(Enum):
    """
    Enum class for the constants used in the Streamlit app.
    """

    #### Main Page ####
    TITLE         = "🔎 HR bot"
    PLACEHOLDER   = "Chi è il miglior candidato per una posizione lavorativa di Sicurezza informatica?"
    MESSAGE_INTRO = "Ciao, Io sono il chatbot che aiuta un HR nei suoi tasks quotidiani. Come posso aiutarti?"
    SPINNER       = "I'm thinking, retriving and elaborating your answer..."
    ERROR_ATT     = "Seleziona il modello da utilizzare dal menù a tendina."
    MODELS        = ("OpenAI", "Llama3")
   

    SUBMIT_BUTTON = "Submit"
    MESSAGE_1 = "Please press the submit button to continue."

