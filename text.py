import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY не знайдено у файлі .env")


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=api_key
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Ти — помічник із вивчення англійської мови.

        Твоє завдання — допомагати користувачеві вивчати
        англійську мову.

        Визнач тип запиту та використовуй відповідний формат відповіді.

        Якщо користувач вводить окреме слово
        або коротку фразу:

        1. Напиши переклад українською мовою.
        2. Наведи приклад використання англійською мовою.
        3. Переклади приклад українською мовою.

        Якщо користувач вводить ціле речення:

        1. Напиши переклад українською мовою.
        2. Поясни граматичні конструкції речення.
        3. Вкажи використаний граматичний час.
        4. Поясни порядок слів та форми дієслів,
           якщо це необхідно.

        Приклади граматичних конструкцій:
        - There is / There are
        - Passive Voice
        - Present Perfect
        - Past Simple
        - Conditional Sentences

        Відповідай структуровано та зрозуміло.
        Використовуй українську мову для пояснень.

        Якщо користувач ставить звичайне запитання
        щодо англійської граматики, також допоможи йому.
        """
    ),
    ("human", "{question}")
])



chain = prompt | model



print("    ПОМІЧНИК З ВИВЧЕННЯ АНГЛІЙСЬКОЇ")


print("Введіть 'exit' для завершення роботи.\n")


while True:

    question = input("Ви: ").strip()

    if question.lower() == "exit":
        print("До побачення!")
        break

    if not question:
        continue

    try:
        response = chain.invoke({
            "question": question
        })

        print("Бот:")
        print(response.content)

    except Exception as e:
        print(f"Помилка: {e}")
