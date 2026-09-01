from langchain_community.tools import DuckDuckGoSearchRun
import dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import re
from langchain_core.tools import tool
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=0.1,
    max_output_tokens=1000
)

# @tool
# def check_password(password: str) -> str:
#     """
#     Перевіряє складність паролю.
#     Перевіряє довжину, наявність літери, цифри,
#     спеціального символу та різні регістри.
#     """
#
#     result = []
#
#     if len(password) > 8:
#         result.append("Добре: пароль має більше 8 символів.")
#     else:
#         result.append("Погано: пароль повинен мати більше 8 символів.")
#
#     if re.search(r"[A-Za-z]", password):
#         result.append("Добре: пароль містить літери.")
#     else:
#         result.append("Погано: пароль не містить літер.")
#
#     if re.search(r"\d", password):
#         result.append("Добре: пароль містить цифру.")
#     else:
#         result.append("Погано: пароль не містить цифр.")
#
#     if re.search(r"[^A-Za-z0-9]", password):
#         result.append("Добре: пароль містить спеціальний символ.")
#     else:
#         result.append("Погано: пароль не містить спеціального символу.")
#
#     if re.search(r"[a-z]", password) and re.search(r"[A-Z]", password):
#         result.append("Добре: пароль містить літери в різних регістрах.")
#     else:
#         result.append(
#             "Погано: пароль повинен містити великі та маленькі літери."
#         )
#
#     return "\n".join(result)
#
#
# tools = [check_password]
#
#
# agent = create_agent(
#     model=llm,
#     tools=tools,
#     system_prompt="""
# Ти агент для перевірки складності паролів.
#
# Якщо користувач надає пароль або просить перевірити пароль,
# використовуй інструмент check_password.
#
# Після отримання результату інструменту поясни користувачу,
# що в його паролі добре, а що потрібно покращити.
# """
# )
#
#
# while True:
#
#     user_input = input("Ви: ")
#
#     if user_input.lower() == "exit":
#         break
#
#     result = agent.invoke({
#         "messages": [
#             {
#                 "role": "user",
#                 "content": user_input
#             }
#         ]
#     })
#
#     print("Бот:", result["messages"][-1].content)


search = DuckDuckGoSearchRun()


agent = create_agent(
    model=llm,
    tools=[search],
    system_prompt="""
Ти агент, який показує останні новини про певну людину.

Користувач повинен вказати ім'я людини.

Якщо користувач вводить ім'я людини:
1. Використай DuckDuckGo для пошуку останніх новин про цю людину.
2. Покажи знайдену інформацію коротко та зрозуміло.
3. Вкажи заголовок або короткий опис новини.
4. Якщо можливо, вкажи дату новини.

Якщо користувач не вводить ім'я людини,
відповідай:

"Немає відповідної інформації."

Не шукай новини, якщо ім'я людини не вказане.
"""
)


while True:

    user_input = input("Ви: ")

    if user_input.lower() == "exit":
        break

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    })

    answer = result["messages"][-1].content

    if isinstance(answer, list):
        answer = answer[0]["text"]

    print("Бот:", answer)