from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


HOSPITAL_DIR = Path("../hospital")


def load_documents():
    workers_path = HOSPITAL_DIR / "For wokers.docx"
    general_path = HOSPITAL_DIR / "General.pdf"

    workers_loader = Docx2txtLoader(str(workers_path))
    general_loader = PyPDFLoader(str(general_path))

    workers_documents = workers_loader.load()
    general_documents = general_loader.load()

    return workers_documents + general_documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return splitter.split_documents(documents)


def add_metadata(chunks):
    current_time = datetime.now().isoformat()

    for i, chunk in enumerate(chunks):
        file_name = Path(chunk.metadata["source"]).name

        chunk.metadata["file_name"] = file_name
        chunk.metadata["chunk_id"] = i
        chunk.metadata["current_time"] = current_time

    return chunks


def main():
    documents = load_documents()

    print(f"Загружено документов/страниц: {len(documents)}")

    chunks = split_documents(documents)

    print(f"Получено chunks: {len(chunks)}")

    chunks = add_metadata(chunks)

    print("\nПервый chunk:")
    print(chunks[0].page_content)

    print("\nMetadata:")
    print(chunks[0].metadata)


if __name__ == "__main__":
    main()