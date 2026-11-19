
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=api_key,
    temperature=0.3,
    timeout=30,
    max_retries=0
)




with open("data/lesson9/return_policy.txt", "r", encoding="utf-8") as file:
    return_policy = file.read()


history = f"""Instruction:
Ти — помічник служби підтримки магазину.
Відповідай на запитання користувачів щодо повернення товарів.
Керуйся правилами повернення, наведеними нижче.
Відповідай українською мовою.
Якщо інформації недостатньо, не вигадуй правила.

Правила повернення товару:
{return_policy}

"""


print("Введіть порожній рядок для завершення діалогу.")


while True:

    question = input("Ви: ").strip()

    if not question:
        print("Діалог завершено.")
        break

    try:
        prompt = history + f"Human: {question}\nAI:"

        response = model.invoke(prompt)

        answer = response.content

        print(f"Бот: {answer}")

        history = prompt + f" {answer}\n"

    except Exception as e:
        print(f"Помилка: {repr(e)}")
