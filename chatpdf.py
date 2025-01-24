import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from googletrans import Translator

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Step 2: Extract text from TXT file
def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

# Step 3: Chunk text
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks

# Step 4: Create embeddings and vector store
@st.cache_resource
def create_embeddings(chunks):
    # Use HuggingFaceEmbeddings for compatibility with local models
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# Step 5: Set up the chat model
def setup_chat_model(vector_store):
    llm = OllamaLLM(model="llama3.2")
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

# Step 6: Multilingual translation
def translate_text(text, target_language="en"):
    translator = Translator()
    return translator.translate(text, dest=target_language).text

def main():
    st.title("Fossong's ChatPDF")
    st.write("Upload a document (PDF or TXT) and ask questions based on its content.")

    # File upload
    uploaded_file = st.file_uploader("Your PDF and TXT files, decoded. Upload once, ask anything", type=["pdf", "txt"])
    user_queries = []

    if uploaded_file:
        with st.spinner("Processing the document..."):
            # Extract text based on file type
            if uploaded_file.name.endswith(".pdf"):
                text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.endswith(".txt"):
                text = extract_text_from_txt(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a PDF or TXT file.")
                return

            # Translate text if not in English
            language = st.selectbox("Select document language", ["English", "French", "Spanish", "German", "Other"])
            if language != "English":
                text = translate_text(text, target_language="en")

            # Chunk and embed text
            chunks = chunk_text(text)
            vector_store = create_embeddings(chunks)
            qa_chain = setup_chat_model(vector_store)

        st.success("Document processed successfully! You can now ask questions.")

        # Question input and response
        question = st.text_input("Ask a question:")
        if question:
            with st.spinner("Analyzing..."):
                user_queries.append(question)
                response = qa_chain({"query": question})  # Pass question as input
                st.write(response["result"])  # Display the result part of the output

                # Optionally display source documents
                # st.subheader("Source Documents")
                # for doc in response["source_documents"]:
                #     st.write(f"- **Source:** {doc.metadata.get('source', 'Original Doc')}")
                #     st.write(doc.page_content)

        # Display logged queries
        if user_queries:
            st.subheader("Query History")
            for i, q in enumerate(user_queries, start=1):
                st.write(f"{i}. {q}")

if __name__ == "__main__":
    main()
