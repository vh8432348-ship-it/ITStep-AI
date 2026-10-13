from langchain_core.tools import tool

from vector_search import search_documents
from database import ask_database


@tool
def search_hospital_documents(query: str) -> str:
    """Шукає інформацію у внутрішніх документах лікарні."""

    results = search_documents(query, top_k=3)

    chunks = []

    for match in results["matches"]:
        metadata = match["metadata"]

        chunks.append(
            f"Файл: {metadata.get('file_name')}\n"
            f"Текст:\n{metadata.get('text', '')}"
        )

    return "\n\n---\n\n".join(chunks)


@tool
def query_hospital_database(query: str) -> str:
    """
    Виконує SELECT-запит до PostgreSQL бази даних лікарні.

    Доступні таблиці:
    departments(id, name)
    doctors(id, name, specialization, department_id)
    patients(id, name, birth_date, phone)
    appointments(id, patient_id, doctor_id, appointment_date, reason)

    Зв'язки:
    doctors.department_id -> departments.id
    appointments.patient_id -> patients.id
    appointments.doctor_id -> doctors.id
    """

    query_clean = query.strip().lower()

    if "sqlite_master" in query_clean:
        return (
            "Не використовуй sqlite_master. "
            "Це PostgreSQL база даних. "
            "Використовуй відомі таблиці: departments, doctors, "
            "patients, appointments."
        )

    if not query_clean.startswith("select"):
        return "Дозволені лише SELECT-запити."

    return str(ask_database(query))