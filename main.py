import os
from langchain_google_genai import ChatGoogleGenerativeAI


os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
)


def zero_shot(topic, audience):
    prompt = f"""
Ти — експерт із розробки навчальних програм.

Створи план навчального курсу.

Тема: {topic}
Цільова аудиторія: {audience}

План повинен містити:
1. Назву курсу.
2. Мету курсу.
3. Очікувані результати навчання.
4. Модулі та теми кожного модуля.
5. Короткий опис тем.
6. Практичні завдання.
7. Фінальний проєкт.
8. Рекомендовану тривалість.

Побудуй курс від простого до складного.
Адаптуй складність матеріалу до цільової аудиторії.
"""

    response = llm.invoke(prompt)
    return response.content


def few_shot(topic, audience):
    prompt = f"""
Ти — експерт із розробки навчальних програм.

Ось приклад:

Тема: Python для початківців
Цільова аудиторія: люди без досвіду програмування.

Назва курсу: Python з нуля

Мета:
Навчити студентів основ програмування мовою Python.

Модуль 1. Основи Python
- Змінні та типи даних
- Введення та виведення
- Арифметичні операції

Модуль 2. Умови та цикли
- if / elif / else
- for
- while

Модуль 3. Колекції
- Списки
- Кортежі
- Множини
- Словники

Фінальний проєкт:
Створення невеликого застосунку.

Очікуваний результат:
Студент може створювати прості програми на Python.

---

Тепер створи новий курс, використовуючи
структуру та підхід із прикладу.

Тема: {topic}
Цільова аудиторія: {audience}

Адаптуй зміст до теми та рівня аудиторії.
Не копіюй зміст прикладу.
"""

    response = llm.invoke(prompt)
    return response.content


while True:

    print("Генератор навчального курсу:")
    print("1 - Zero-shot")
    print("2 - Few-shot")
    print("0 - Вихід")

    choice = input("\nОберіть режим: ")

    if choice == "0":
        print("Програму завершено.")
        break

    if choice not in ["1", "2"]:
        print("Невірний вибір.")
        continue

    topic = input("Введіть тему курсу: ")
    audience = input("Опишіть цільову аудиторію: ")

    print("Генерація курсу..")

    if choice == "1":
        result = zero_shot(topic, audience)
    else:
        result = few_shot(topic, audience)

    print(result)