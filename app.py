import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Page Config
st.set_page_config(page_title="The Eternal Archive", page_icon="🏛️")

st.title("🏛️ The Eternal Archive")
st.markdown("""
*Welcome to the ghostly library. Upload documents to animate their spirits and converse with the collective knowledge of the archive.*
""")

# Sidebar for Configuration
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    st.info("Get a free key at [console.groq.com](https://console.groq.com)")
    
    uploaded_files = st.file_uploader("Upload Documents (PDF or TXT)", 
                                     type=["pdf", "txt"], 
                                     accept_multiple_files=True)

# Initialize Session State
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process Documents
if uploaded_files and groq_api_key:
    if st.button("Animate Archive"):
        with st.spinner("Awakening the spirits..."):
            try:
                all_docs = []
                # Save files temporarily to load them
                if not os.path.exists("temp"):
                    os.makedirs("temp")
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("temp", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                    else:
                        loader = TextLoader(file_path)
                    all_docs.extend(loader.load())
                
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                splits = text_splitter.split_documents(all_docs)
                
                # Create Embeddings using open-source HuggingFace model
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Build FAISS Vector Store
                vector_db = FAISS.from_documents(splits, embeddings)
                st.session_state.vector_db = vector_db
                st.success("The archive is now alive!")
            except Exception as e:
                st.error(f"Error awakening archive: {e}")

# Chat Interface
if st.session_state.vector_db and groq_api_key:
    # Initialize LLM (Llama 3 via Groq)
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
    
    # Create RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.vector_db.as_retriever(),
    )

    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask the archive..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting the spirits..."):
                response = qa_chain.invoke(prompt)
                answer = response["result"]
                st.markdown(answer)
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
elif not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar to begin.")
elif not st.session_state.vector_db:
    st.info("Upload documents and click 'Animate Archive' to start the séance.")

