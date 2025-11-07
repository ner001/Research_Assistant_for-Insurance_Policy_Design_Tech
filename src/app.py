import streamlit as st
from agents.analyzer_agent import DocumentAnalyzerAgent
from components.document_uploader import upload_document

# Initialize the summarizer agent
st.title("Document Analysis")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save the uploaded file temporarily
    with open("uploaded_document.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Initialize the summarizer agent
    agent = DocumentAnalyzerAgent(pdf_path="uploaded_document.pdf")
    
    # Process the document
    st.write("Processing the document...")
    agent.process_document()
    st.success("Document processed successfully!")
    
    # Query input
    query = st.text_input("Enter your query:")
    if query:
        answer = agent.answer_query(query)
        st.write("Answer:")
        st.write(answer)
        st.write("---") 
        
