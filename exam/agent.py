import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from tools import search_hospital_documents, query_hospital_database


load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

tools = [
    search_hospital_documents,
    query_hospital_database,
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
    Ти — чат-бот медичного центру.

    Ти маєш доступ до двох інструментів:

    1. search_hospital_documents —
       пошук інформації у внутрішніх документах медичного центру.

    2. query_hospital_database —
       отримання даних із PostgreSQL бази даних медичного центру.

    Для бази даних тобі НЕ потрібно отримувати схему через SQL.
    Схема вже відома:

    departments:
    - id
    - name

    doctors:
    - id
    - name
    - specialization
    - department_id

    patients:
    - id
    - name
    - birth_date
    - phone

    appointments:
    - id
    - patient_id
    - doctor_id
    - appointment_date
    - reason

    Зв'язки:
    - doctors.department_id = departments.id
    - appointments.patient_id = patients.id
    - appointments.doctor_id = doctors.id

    НІКОЛИ не використовуй sqlite_master.

    Використовуй search_hospital_documents, якщо користувач
    запитує про правила, обов'язки працівників, протоколи,
    процедури або іншу інформацію з документів.

    Використовуй query_hospital_database, якщо користувач
    запитує про пацієнтів, лікарів, відділення або записи
    на прийом.

    Якщо для відповіді потрібна інформація з обох джерел,
    використовуй обидва інструменти.

    Не вигадуй інформацію, якої немає в документах або базі даних.

    Відповідай українською мовою.

    Відповідай коротко та зрозуміло.
    """,
)

