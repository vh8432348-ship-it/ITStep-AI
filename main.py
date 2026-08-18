import dotenv
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")



llm = GoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=0.1,
    max_output_tokens=1000
)


# zero_shot_prompt = PromptTemplate(
#     input_variables=["language", "task"],
#     template="""
# Ти -- програміст.
#
# ###ІНСТРУКЦІЯ###
#
# 1. **Пиши код мовою програмування, яку вказав користувач.**
# 2. **Уважно проаналізуй завдання.**
# 3. **Напиши код, який вирішує поставлену задачу.**
# 4. **Поверни тільки код без пояснень.**
#
# ###МОВА ПРОГРАМУВАННЯ###
#
# {language}
#
# ###ЗАВДАННЯ###
#
# {task}
#
# ###ВІДПОВІДЬ###
# """
# )
#
#
# language = input("Мова програмування: ")
# task = input("Опис задачі: ")
#
# prompt = zero_shot_prompt.format(
#     language=language,
#     task=task
# )
#
# result = llm.invoke(prompt)
#
# print(result)
# #/////////
# few_shot_prompt = PromptTemplate(
#     input_variables=["language", "task"],
#     template="""
# Ти -- програміст, який спеціалізується на написанні функцій.
# Твоя задача -- написати код для вирішення задачі користувача.
#
# ###ІНСТРУКЦІЯ###
#
# 1. **Пиши код мовою програмування, яку вказав користувач.**
# 2. **Поверни тільки код без пояснень.**
# 3. **Не використовуй Markdown та блоки ``` ```.**
# 4. **Результат повинен бути готовою функцією, яка вирішує поставлену задачу.**
# 5. **Дотримуйся стилю коду, показаного у прикладах.**
# 6. **Не копіюй код із прикладів, якщо він не підходить для нової задачі.**
# 7. **Самостійно адаптуй підхід із прикладів до нової задачі.**
#
# ###ПРИКЛАД 1###
#
# **Мова програмування:** Python
#
# **Завдання:** Написати функцію, яка повертає суму двох чисел.
#
# **Відповідь:**
#
# def add(a, b):
#     return a + b
#
#
# ###ПРИКЛАД 2###
#
# **Мова програмування:** Python
#
# **Завдання:** Написати функцію, яка перевіряє, чи є число парним.
#
# **Відповідь:**
#
# def is_even(number):
#     return number % 2 == 0
#
#
# ###НОВА ЗАДАЧА###
#
# **Мова програмування:**
# {language}
#
# **Опис задачі:**
# {task}
#
# ###ВІДПОВІДЬ###
# """
# )
#
# language = input("Мова програмування: ")
# task = input("Опис задачі: ")
#
# prompt = few_shot_prompt.format(
#     language=language,
#     task=task
# )
#
# result = llm.invoke(prompt)
#
# print(result)

# Завдання 2
zero_shot_prompt = PromptTemplate(
    input_variables=["user_text"],
    template=""" Тобі потрібно передодити текст з неформального стилю в фармальний.
    ###ІНСТРУКЦІЯ###
1. **Не пиши нічого окрім переведенного тексту користувача.**
2. **Виправ орфогарафічні помилки в тексті.**
**Текст користувача:**
{user_text}
"""
)
user_text = input("Введіть не формальний текст для переведення его в формальний")
prompt = zero_shot_prompt.format(
    user_text=user_text
)

result = llm.invoke(prompt)

print(result)

few_shot_prompt = PromptTemplate(
    input_variables=["user_text"],
    template=""" Тобі потрібно передодити текст з неформального стилю в фармальний.
    ###ІНСТРУКЦІЯ###
1. **Не пиши нічого окрім переведенного тексту користувача.**
2. **Виправ орфогарафічні помилки в тексті.**
###ПРИКЛАД 1###

 **Текст користувача:**
  Привіт! Я хотів спитати, чи можеш ти скинути мені той файл сьогодні? 
  Бо він мені потрібен для роботи, а я щось зовсім забув його завантажити.

 **Відповідь:**
    Доброго дня! Хотів би уточнити, чи могли б Ви надіслати мені цей файл сьогодні? 
    Він необхідний мені для роботи, оскільки я забув завантажити його раніше.


**Текст користувача:**
{user_text}
"""
)
user_text = input("Введіть не формальний текст для переведення его в формальний")
prompt = zero_shot_prompt.format(
    user_text=user_text
)

result = llm.invoke(prompt)

print(result)