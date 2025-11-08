from utils.text_extractor import extract_text_with_pypdf2
from utils.text_cleaner import clean_text
from utils.chunker import chunk_text, enrich_chunk_with_llama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def process_document(pdf_path, doc_id="insurance_doc_001", llm=None, max_workers=4):
    """Complete pipeline: extract, chunk, enrich (parallelized), and store in vector DB

    Args:
        pdf_path (str): path to the PDF file
        doc_id (str): document identifier used in metadata
        llm: optional LLM object to pass into enrich_chunk_with_llama
        max_workers (int): number of threads for parallel enrichment

    Returns:
        tuple: (vectorstore, metadatas)
    """

    # Step 1: Extract text
    full_text = extract_text_with_pypdf2(pdf_path)

    # Step 2: Clean text
    cleaned_text = clean_text(full_text)

    # Show preprocessing stats
    original_chars = len(full_text)
    cleaned_chars = len(cleaned_text)
    reduction_percent = ((original_chars - cleaned_chars) / original_chars) * 100 if original_chars else 0
    print(f"   Original: {original_chars:,} characters")
    print(f"   Cleaned: {cleaned_chars:,} characters")
    print(f"   Reduction: {reduction_percent:.1f}%")

    # Step 3: Chunk the text
    chunks = chunk_text(cleaned_text, chunk_size=2000, chunk_overlap=200)

    start_time = time.time()
    texts_for_embedding = []
    metadata_list = []

    print(f"🚀 Enriching {len(chunks)} chunks in parallel...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(enrich_chunk_with_llama, chunk, llm): idx
            for idx, chunk in enumerate(chunks)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                enrichment = future.result()
            except Exception as e:
                print(f"⚠️ Error processing chunk {idx}: {e}")
                enrichment = {
                    "summary": "Summary not available",
                    "keywords": "No keywords extracted",
                    "questions": "No questions generated"
                }

            chunk = chunks[idx]

            # Metadata dict for this chunk
            meta = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{idx}",
                "chunk_index": idx,
                "original_text": chunk,
                "summary": enrichment.get("summary"),
                "keywords": enrichment.get("keywords"),
                "questions": enrichment.get("questions")
            }

            # Text to embed: combine summary + keywords + questions + original_text
            text_for_embedding = (
                f"Summary: {enrichment.get('summary')}\n"
                f"Keywords: {enrichment.get('keywords')}\n"
                f"Questions: {enrichment.get('questions')}\n"
                f"Original Text: {chunk}"
            )

            texts_for_embedding.append(text_for_embedding)
            metadata_list.append(meta)

    end_time = time.time()
    enrichment_time = end_time - start_time
    print(f"🏁 Parallel enrichment completed in {enrichment_time:.2f} seconds")

    # Step 4: Create embeddings and FAISS vector store
    print("\n🔍 Creating vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_texts(
        texts=texts_for_embedding,
        embedding=embeddings,
        metadatas=metadata_list
    )

    print("✅ Vector store created successfully!")
    return vectorstore, metadata_list
