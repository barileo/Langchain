import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


def main():
    print("Hello from langchain-course!")
    print(f"PINE API {os.environ['PINECONE_API_KEY']}")
    loader = TextLoader(
        "/Users/barinderpaulsingh/Projects/Udemy LangChain Agentic AI/langchain-course/mediumblog1.txt"
    )

    print(f"Loading data....")
    document = loader.load()

    print(f"Splitting data....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents=document)
    print(f"Chunks created {len(texts)}")

    print(f"Creating embeddings....")
    embedding = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))

    print(f"Uploading to Pinecone vector store ....")
    PineconeVectorStore.from_documents(
        texts, embedding=embedding, index_name=os.environ.get("INDEX")
    )

    print("Finish")


if __name__ == "__main__":
    main()
