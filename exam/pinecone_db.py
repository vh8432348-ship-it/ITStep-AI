import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore

from document import load_documents, split_documents, add_metadata


load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

index_name = "hospital-documents"

# Удаляем старый индекс, созданный под OpenAI embeddings
if pc.has_index(index_name):
    pc.delete_index(index_name)

# Загружаем локальную модель
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# У этой модели размер вектора = 384
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1",
    ),
)

index = pc.Index(index_name)


class LocalEmbeddings:
    def embed_documents(self, texts):
        return embedding_model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()

    def embed_query(self, text):
        return embedding_model.encode(
            text,
            normalize_embeddings=True
        ).tolist()


embeddings = LocalEmbeddings()

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
)

documents = load_documents()
chunks = split_documents(documents)
chunks = add_metadata(chunks)

vector_store.add_documents(chunks)

print(f"Загружено chunks в Pinecone: {len(chunks)}")