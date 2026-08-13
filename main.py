# завантеження api key як змінну середовища
import dotenv
import os
from langchain_google_genai import GoogleGenerativeAI

# завантадити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")



llm = GoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=0.1,
    max_output_tokens=100
)


# result = llm.invoke(
#     "Напиши коротку історію. "
# )
#
# print(result)



history = ''
count1 = 0
count = 0
while True:
    question = input("Ваше питання: ")

    if question == '':
        break

    prompt = f"""
    Ти повинен відповідати на питання користувача, як Брет Піт
    

    Питання користувача:
    {question}
    Історія вашого спілкування:
    {history}
    Відповідай коротко та зрозуміло.
    """
    result = llm.invoke(prompt)
    count += 1
    count1 += 1
    history += f"Користувач питання {count}:{question}. Бот відповідь: {count1} {result}."
    print(result)