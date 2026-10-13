import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase


load_dotenv()

user = os.getenv("DB_USER")
password = quote_plus(os.getenv("DB_PASSWORD"))
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
database = os.getenv("DB_NAME")

db_uri = (
    f"postgresql+psycopg2://{user}:{password}"
    f"@{host}:{port}/{database}"
)

db = SQLDatabase.from_uri(db_uri)

engine = create_engine(db_uri)


def ask_database(query: str):
    with engine.connect() as connection:
        result = connection.execute(text(query))
        return result.fetchall()


if __name__ == "__main__":
    print("Підключення до Supabase через LangChain успішно!")
    print("Таблиці:", db.get_usable_table_names())

    result = ask_database(
        """
        SELECT
            p.name AS patient,
            d.name AS doctor,
            a.appointment_date,
            a.reason
        FROM appointments a
        JOIN patients p ON a.patient_id = p.id
        JOIN doctors d ON a.doctor_id = d.id;
        """
    )

    print(result)