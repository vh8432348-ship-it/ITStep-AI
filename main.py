import os
import json
import uuid

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()


google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")



pc = Pinecone(api_key=pinecone_api_key)

index_name = "lesson-rag"



if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )



embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=google_api_key
)



folder_path = "data/lesson_rag/files"

documents = []
ids = []
file_ids = {}



for filename in os.listdir(folder_path):

    file_path = os.path.join(folder_path, filename)

    if not os.path.isfile(file_path):
        continue

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    document_id = str(uuid.uuid4())

    document = Document(
        page_content=content,
        metadata={
            "path": file_path
        }
    )

    documents.append(document)
    ids.append(document_id)

    file_ids[document_id] = filename



with open(
    "data/lesson_rag/file_ids.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        file_ids,
        file,
        ensure_ascii=False,
        indent=4
    )




vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
    pinecone_api_key=pinecone_api_key
)



vectorstore.add_documents(
    documents=documents,
    ids=ids
)


print("Векторна база даних створена.")
print(f"Додано документів: {len(documents)}")
print("ID збережені у file_ids.json")


query = input("\nВведіть запит для пошуку: ")

results = vectorstore.similarity_search(
    query,
    k=3
)


print("Результати пошуку:")

for result in results:

    print("Файл:", result.metadata["path"])
    print("Вміст:")
    print(result.page_content[:500])