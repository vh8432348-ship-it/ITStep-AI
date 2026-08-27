from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
from langchain_core.messages import trim_messages

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")



llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=0.1,
    max_output_tokens=1000
)


with open("data/lesson9/return_policy.txt", "r", encoding="utf-8") as file:
    rules = file.read()


# Інструкція для моделі
messages = [
    SystemMessage(content=f"""
Ти чат-бот, який відповідає тільки на питання
стосовно умов повернення товару.

Використовуй інформацію тільки з наведених правил:

{rules}

Якщо користувач запитує щось, що не стосується
умов повернення товару або на це немає інформації
у правилах, відповідай:

"Немає інформації з цього питання."
""")
]


trimmer = trim_messages(
    max_tokens=5,
    strategy="last",
    token_counter=len,
    start_on="human",
    end_on="human",
    include_system=True
)

#
# while True:
#
#     user_input = input("Ви: ")
#
#     if user_input.lower() == "exit":
#         break
#
#     messages.append(
#         HumanMessage(content=user_input)
#     )
#
#     trimmed_messages = trimmer.invoke(messages)
#
#     response = llm.invoke(trimmed_messages)
#
#     print("Бот:", response.content[0]["text"])
#
#     messages.append(
#         AIMessage(content=response.content)
#     )
#

messages = [
    SystemMessage(content="""
Ти допомагаєш користувачу вивчати англійську мову.

Твої правила:

1. Якщо користувач просить перекласти англійське слово
або коротку фразу:
- дай переклад;
- наведи приклад використання цього слова або фрази
  в англійському реченні;
- переклади приклад українською.

2. Якщо користувач просить перекласти ціле речення:
- спочатку дай переклад речення;
- потім поясни граматику речення;
- зверни увагу на граматичні конструкції,
  час, порядок слів, утворення питання тощо.

Відповідай українською мовою.
"""),

    HumanMessage(content="Переклади слово beautiful"),

    AIMessage(content="""
Переклад: beautiful — красивий, прекрасний.

Приклад:
She has a beautiful smile.
— У неї прекрасна посмішка.
"""),

    # Приклад 2
    HumanMessage(content="Переклади речення There is a book on the table."),

    AIMessage(content="""
Переклад: На столі є книга.

Граматика:
There is використовується, коли ми говоримо про
наявність одного предмета або особи.

There is + однина:
There is a book on the table.

Для множини використовується There are:
There are two books on the table.

Структура:
There is/are + предмет + місце.
""")
]


while True:

    user_input = input("Ви: ")

    if user_input.lower() == "exit":
        break

    messages.append(
        HumanMessage(content=user_input)
    )

    response = llm.invoke(messages)

    if isinstance(response.content, list):
        answer = response.content[0]["text"]
    else:
        answer = response.content

    print("Бот:", answer)

    messages.append(
        AIMessage(content=answer)
    )