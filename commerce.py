"""
DoC Agentic AI Assistant
Stack: OpenAI embeddings + GPT-4 + Pinecone vectorstore + LangChain ConversationalRetrievalChain

Install deps:
    pip install langchain langchain-community langchain-openai langchain-pinecone \
                pinecone-client python-dotenv unstructured pypdf

Set env vars (or create a .env file):
    OPENAI_API_KEY=sk-...
    PINECONE_API_KEY=...
    PINECONE_INDEX=commerce-docs      # must exist in your Pinecone project
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

load_dotenv()

OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "commerce-docs")
DOCS_DIR         = "./docs"   # folder containing your PDFs / text files


# ── 1. Ingest & embed documents ───────────────────────────────────────────────
def build_vectorstore(docs_dir: str) -> PineconeVectorStore:
    """Load docs, chunk them, embed and upsert into Pinecone."""
    print("Loading documents...")
    loader = DirectoryLoader(docs_dir, glob="**/*.pdf")
    docs = loader.load()
    if not docs:
        raise FileNotFoundError(f"No PDFs found in '{docs_dir}'")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"  {len(docs)} docs → {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",   # or text-embedding-ada-002
        api_key=OPENAI_API_KEY,
    )

    print("Upserting to Pinecone...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX,
    )
    print("  Done.")
    return vectorstore


def load_vectorstore() -> PineconeVectorStore:
    """Connect to an already-populated Pinecone index (skip re-ingestion)."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )
    return PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
    )


# ── 2. Build the chain ────────────────────────────────────────────────────────
def build_chain(vectorstore: PineconeVectorStore) -> ConversationalRetrievalChain:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    QA_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant for U.S. Department of Commerce staff.
Answer using only the context below. Be concise and professional.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
Answer:""",
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=False,
    )
    return chain


# ── 3. Interactive REPL ───────────────────────────────────────────────────────
def run(chain: ConversationalRetrievalChain) -> None:
    print("\nAssistant ready. Type 'quit' to exit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("quit", "exit"):
            break

        result = chain.invoke({"question": q})
        print(f"\nAssistant: {result['answer']}")

        sources = list({
            d.metadata.get("source", "unknown")
            for d in result.get("source_documents", [])
        })
        if sources:
            print(f"Sources: {', '.join(sources)}")
        print()


# ── 4. Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Pass --ingest flag the first time to load docs into Pinecone.
    # After that, run without the flag to skip re-ingestion.
    if "--ingest" in sys.argv:
        vs = build_vectorstore(DOCS_DIR)
    else:
        vs = load_vectorstore()

    chain = build_chain(vs)
    run(chain)