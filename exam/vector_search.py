import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

index = pc.Index("hospital-documents")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def search_documents(query: str, top_k: int = 3):
    query_vector = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    return result


if __name__ == "__main__":
    results = search_documents(
        "Які зобовязаності є у робітників лікарні?"
    )

    for match in results["matches"]:
        print("\n---")
        print("Score:", match["score"])
        print("File:", match["metadata"].get("file_name"))
        print("Text:")
        print(match["metadata"].get("text"))