from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
#from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# snippet: returning sources
from langchain_core.runnables import RunnableLambda

#https://blog.sudhirnakka.com/2025/build_rag_ollama

DATA_DIR = Path("docs")
DB_DIR = "./chroma_db"

# 1) Load local documents
def load_docs():
    docs = []
    # Local pdf docs
    for p in DATA_DIR.glob("**/*"):
        if p.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
    return docs

# 2) Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256, chunk_overlap=50, add_start_index=True
)

# 3) Embeddings via Ollama
# Choose your pulled embedding model name
EMBED_MODEL = "nomic-embed-text"  # or "mxbai-embed-large"
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# 4) Vector store (Chroma)
vectorstore = Chroma(
    collection_name="local_rag", embedding_function=embeddings, persist_directory=DB_DIR
)

# 5) Indexing function

def build_or_load_index():
    # If DB exists, you can skip re-index; here we rebuild for demo
    docs = load_docs()
    splits = text_splitter.split_documents(docs)

    #vectorstore.reset_collection()

    # Recreate collection for a fresh demo
    #vectorstore.delete_collection()
    vectorstore.add_documents(splits)
    #vectorstore.persist()

# 6) Retriever
retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 7})

# 7) LLM via Ollama
CHAT_MODEL = "llama3.1:8b"
#llm = Ollama(model=CHAT_MODEL)
llm = OllamaLLM(model=CHAT_MODEL)

# 8) Prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. Answer the user question and provide citations using the provided context.
    If the answer is not in the context, say you don't know.

    Remember, you must return both an answer and citations. A citation consists of a VERBATIM quote that \
    justifies the answer and the ID of the quote article. Return a citation for every quote across all articles \
    that justify the answer. Use the following format for your final output:
    
    <cited_answer>
        <answer></answer>
        <citations>
            <citation><source_id></source_id><quote></quote></citation>
            <citation><source_id></source_id><quote></quote></citation>
            ...
        </citations>
    </cited_answer>

    Context:{context}

    Question: {question}
    """
)

# 9) Chain: retrieve -> prompt -> llm -> parse
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# 10) citations
def format_docs(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source") or meta.get("file_path") or "unknown"
        parts.append(f"[{i}] {src}:\n{d.page_content[:500]}\n")
    return "\n".join(parts)

retrieve = RunnableLambda(lambda q: retriever.invoke(q))
chain_with_sources = (
    {"context": retrieve | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    # First time: build the index
    if not Path(DB_DIR).exists():
        Path(DB_DIR).mkdir(parents=True, exist_ok=True)
    build_or_load_index()

    # Simple CLI
    print("RAG ready. Ask a question (Ctrl+C to exit).")
    while True:
        try:
            q = input("\nYou: ")
            if not q.strip():
                continue
            #answer = rag_chain.invoke(q)
            answer = chain_with_sources.invoke(q)
            print("\nAssistant:\n", answer)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break