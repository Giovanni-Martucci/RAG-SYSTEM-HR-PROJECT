import streamlit as st
import time
import os
import requests
from utils.rag_memory_system import rag_system
from utils.constants import StreamlitConstants
from aiohttp import ClientSession


st.title(StreamlitConstants.TITLE.value)

with st.sidebar:

    model = st.selectbox(
        "-1-. Choose the model:",
        StreamlitConstants.MODELS.value
    )

    if "OpenAI" in model:
        openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")
    else:
        openai_api_key = None
        

    if st.button('Confirm'):
        if "rag_tool" not in st.session_state:
            alert = st.warning(f"Hai selezionato {model}!") 
            time.sleep(2) 
            alert.empty()
            st.session_state.rag_tool = rag_system(openai_api_key=openai_api_key)
            

    uploaded_files = st.file_uploader("\n-2-. Choose a PDF curriculum files", accept_multiple_files=True, type="pdf")
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        document_path = os.path.join("./documents", uploaded_file.name)
        with open(document_path, 'wb') as f: 
            f.write(bytes_data) 
    
    if len(uploaded_files):
        st.session_state.rag_tool.setup_documents()
               
    with st.form("my_form"): 
        st.write("List of curriculum files")
        list_of_del_file = []        
        for count, file in enumerate(os.listdir("./documents")):
            if file != ".DS_Store":
                key = f'chkbox_{count}'
                checkbox = st.checkbox(file, key=key)
                list_of_del_file.append((checkbox,key,file))
                
                            
        submitted = st.form_submit_button("Delete")
        if submitted:
            for checkbox, key, file in list_of_del_file:
                if checkbox:
                    os.remove(os.path.join("./documents", file))
                    del st.session_state[key]
            
                
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": StreamlitConstants.MESSAGE_INTRO.value}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder=StreamlitConstants.PLACEHOLDER.value):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    
    with st.chat_message("assistant"):
        with st.spinner(StreamlitConstants.SPINNER.value):
            try:
                response = st.session_state.rag_tool.ask(prompt, "sistemadisessionid_108hgfht31bkjkj9380918")
                st.session_state.messages.append({"role": "assistant", "content": response})
            except AttributeError:
                response = StreamlitConstants.ERROR_ATT.value     
        st.write(response)
