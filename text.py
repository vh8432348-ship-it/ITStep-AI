
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")



model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=api_key,
    temperature=0.3,
    timeout=30,
    max_retries=0
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Ти — помічник із вивчення англійської мови.

        Твоє завдання — допомагати користувачеві
        вивчати англійську мову.

        Якщо користувач вітається, прощається
        або просто спілкується з тобою:
        - Відповідай природно та дружелюбно.
        - Не перекладай такі повідомлення.
        - Підтримуй діалог.

        Якщо користувач вводить окреме слово
        або коротку фразу:
        1. Напиши переклад українською мовою.
        2. Наведи приклад використання англійською.
        3. Переклади приклад українською.

        Якщо користувач вводить ціле речення:
        1. Напиши переклад українською мовою.
        2. Поясни граматичні конструкції речення.
        3. Вкажи використаний граматичний час.
        4. Поясни порядок слів та форми дієслів.

        Якщо користувач ставить запитання
        щодо англійської граматики:
        - Дай зрозуміле пояснення.
        - Наведи приклади.

        Завжди використовуй українську мову
        для відповідей.
        """
    ),
    ("human", "{question}")
])


chain = prompt | model


print("     ПОМІЧНИК З ВИВЧЕННЯ АНГЛІЙСЬКОЇ")

print("Введіть 'exit' для завершення роботи")


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
        print(response.text)

    except Exception as e:
        print(f"Помилка: {repr(e)}")
