import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
)


parser = StrOutputParser()


exercise_prompt = PromptTemplate(
    template="""
    Ти професійний тренер.

    Користувач хоче отримати план тренувань.
    Його мета: {goal}

    Склади список з 8-10 вправ, які підходять для цієї мети.

    Поверни тільки список вправ з коротким описом.
    Не створюй розклад тренувань.
    """,
    input_variables=["goal"]
)

exercise_chain = exercise_prompt | llm | parser


plan_prompt = PromptTemplate(
    template="""
    Ти професійний тренер.

    Ось список вправ:
    {exercises}

    Рівень підготовки користувача:
    {level}

    Кількість часу, яку користувач може витрачати
    на тренування протягом тижня:
    {hours} годин.

    На основі цих даних склади персональний план тренувань.

    Вкажи:
    - кількість тренувань на тиждень;
    - тривалість кожного тренування;
    - які вправи виконувати;
    - кількість підходів і повторень;
    - час відпочинку;
    - як розподілити тренування протягом тижня.

    План повинен відповідати рівню підготовки
    та доступній кількості часу.
    """,
    input_variables=["exercises", "level", "hours"]
)

plan_chain = plan_prompt | llm | parser


goal = input("Мета тренування: ")
level = input("Рівень підготовки (низький/середній/професіонал): ")
hours = input("Кількість годин на тиждень: ")


exercises = exercise_chain.invoke({
    "goal": goal
})

print("Список вправ:")
print(exercises)


plan = plan_chain.invoke({
    "exercises": exercises,
    "level": level,
    "hours": hours
})

print("План тренувань:")
print(plan)
