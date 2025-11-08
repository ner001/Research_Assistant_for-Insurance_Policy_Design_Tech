from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama


def chunk_text(text: str, chunk_size=2500, chunk_overlap=300):
    """Chunk text using sentence splitter"""
    document = Document(text=text)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents([document])
    chunks = [node.text for node in nodes]
    print(f"Created {len(chunks)} chunks")
    return chunks

# try to create a default module-level LLM, but don't crash if unavailable
try:
    _module_llm = Ollama(model="llama3:8b", temperature=0.3)
except Exception:
    _module_llm = None


def enrich_chunk_with_llama(chunk, llm=None):
    """Use LLaMA to generate all enrichments for a single chunk.
    Accepts an optional llm; if not provided, uses module-level LLM or a safe stub.
    """
    # select LLM: prefer passed llm, then module-level, then attempt to instantiate, else stub
    llm_to_use = llm or _module_llm
    if llm_to_use is None:
        try:
            llm_to_use = Ollama(model="llama3:8b", temperature=0.3)
        except Exception:
            class _StubLLM:
                def invoke(self, prompt: str):
                    return (
                        "SUMMARY: Summary not available\n"
                        "KEYWORDS: No keywords extracted\n"
                        "QUESTIONS: No questions generated"
                    )
            llm_to_use = _StubLLM()

    combined_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Analyze the following document text and provide:

{text}

Please respond in this exact format:
SUMMARY: [Summarize the following insurance text in 2-3 sentences, focusing on key points and main topics]
KEYWORDS: [Extract 5-7 key insurance terms and concepts from this text. Return only the keywords separated by commas]
QUESTIONS: [Generate 3 hypothetical questions that this insurance text could answer. Return questions separated by newlines]

Response:"""
    )

    try:
        # build prompt string and call LLM
        prompt_str = combined_prompt.format(text=chunk)
        response = llm_to_use.invoke(prompt_str)
        response_text = response.strip() if hasattr(response, 'strip') else str(response).strip()

        # Parse the combined response
        summary = ""
        keywords = ""
        questions = ""

        lines = response_text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('SUMMARY:'):
                current_section = 'summary'
                summary = line.replace('SUMMARY:', '').strip()
            elif line.startswith('KEYWORDS:'):
                current_section = 'keywords'
                keywords = line.replace('KEYWORDS:', '').strip()
            elif line.startswith('QUESTIONS:'):
                current_section = 'questions'
                questions = line.replace('QUESTIONS:', '').strip()
            elif current_section and line:
                if current_section == 'summary':
                    summary += " " + line
                elif current_section == 'keywords':
                    keywords += " " + line
                elif current_section == 'questions':
                    questions += "\n" + line

        return {
            "summary": summary.strip() or "Summary not available",
            "keywords": keywords.strip() or "No keywords extracted",
            "questions": questions.strip() or "No questions generated"
        }

    except Exception as e:
        print(f"Error enriching chunk: {e}")
        return {
            "summary": "Summary not available",
            "keywords": "No keywords extracted",
            "questions": "No questions generated"
        }