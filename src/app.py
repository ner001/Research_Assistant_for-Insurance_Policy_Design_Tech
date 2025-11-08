import streamlit as st
from agents.analyzer_agent import DocumentAnalyzerAgent
from agents.summarizer_agent import SummarizerAgent
from components.document_uploader import upload_document
from utils.process_document import process_document
from mcp import Context

st.title("Document Analysis")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # save file once
    path = "uploaded_document.pdf"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # only process once per uploaded file; keep vectorstore in session_state
    processed_key = f"processed::{uploaded_file.name}::{uploaded_file.size}"
    if processed_key not in st.session_state:
        st.info("Processing the document...")
        vectorstore, metadatas = process_document(path)
        st.session_state[processed_key] = {
            "vectorstore": vectorstore,
            "metadatas": metadatas
        }
        st.success("Document processed successfully!")
    else:
        stored = st.session_state[processed_key]
        vectorstore, metadatas = stored["vectorstore"], stored["metadatas"]

    # populate MCP context with per-chunk summaries + metadata for other agents
    chunk_summaries = [md.get("summary") for md in metadatas if isinstance(md, dict) and md.get("summary")]

    # create and populate shared Context
    context = Context()
    context.set("chunk_summaries", chunk_summaries)
    context.set("metadatas", metadatas)

    # instantiate agents with shared context
    agent = DocumentAnalyzerAgent(vectorstore=vectorstore, metadatas=metadatas, context=context)
    summarizer = SummarizerAgent(vectorstore=vectorstore, metadatas=metadatas, context=context)
    
    query = st.text_input("Enter your query:")
    if query:
        answer = agent.answer_query(query)
        st.write("Answer:")
        st.write(answer)
        st.write("---")

    # Summarization button -> use the per-chunk summaries in context / metadatas
    if st.button("Generate Global Summary"):
        with st.spinner("Generating global summary..."):
            # summarizer will read chunk_summaries from context if none passed
            summary = summarizer.generate_summary()
        st.subheader("Global Summary")
        st.write(summary)

