import streamlit as st
from google import genai


api_key = st.secrets["GEMINI_API_KEY"]

client = genai.Client(api_key=api_key)


st.title("Спілкування з відомою людиною")


person = st.text_input("З ким ви хочете поспілкуватися?")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if person:

    prompt = st.chat_input("Напишіть повідомлення...")

    if prompt:

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": f"""
                        Ти симулюєш спілкування з відомою людиною: {person}.

                        Відповідай так, ніби ти ця людина.
                        Намагайся враховувати її відомі погляди,
                        професію, стиль спілкування та біографію.

                        Не кажи, що ти справжня ця людина.
                        Це лише рольова симуляція.
                        """
                    }
                ]
            }
        ]

        for message in st.session_state.messages:

            role = "model" if message["role"] == "assistant" else "user"

            contents.append({
                "role": role,
                "parts": [
                    {
                        "text": message["content"]
                    }
                ]
            })

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=contents
        )

        answer = response.text

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.markdown(answer)

# Задання 2

# api_key = st.secrets["GEMINI_API_KEY"]
# client = genai.Client(api_key=api_key)
#
#
# st.title("Симулятор співбесіди")
#
#
# position = st.text_input("Введіть назву посади:")
#
#
# job_file = st.file_uploader(
#     "Завантажте опис вакансії:",
#     type=["txt", "pdf", "docx"]
# )
#
#
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#
#
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])
#
#
# if position and job_file:
#
#     if job_file.type == "text/plain":
#         job_description = job_file.read().decode("utf-8")
#
#     else:
#         job_description = (
#             "Опис вакансії завантажено у файлі. "
#             "Використовуй інформацію з назви посади "
#             "та доступного контексту вакансії."
#         )
#
#
#     prompt = st.chat_input("Ваша відповідь на співбесіді:")
#
#
#     if prompt:
#
#         st.session_state.messages.append({
#             "role": "user",
#             "content": prompt
#         })
#
#
#         contents = [
#             {
#                 "role": "user",
#                 "parts": [
#                     {
#                         "text": f"""
# Ти симулюєш співбесіду на посаду "{position}".
#
# Ти виступаєш у ролі професійного HR-інтерв'юера.
#
# Опис вакансії:
# {job_description}
#
# Проводь співбесіду відповідно до цієї вакансії.
# Став по одному питанню за раз.
# Оцінюй відповіді кандидата.
# Поступово переходь від загальних питань до технічних
# та питань, пов'язаних з конкретною посадою.
#
# Не давай правильну відповідь замість кандидата.
# Після кожної відповіді став наступне питання.
# """
#                     }
#                 ]
#             }
#         ]
#
#
#         for message in st.session_state.messages:
#             role = "model" if message["role"] == "assistant" else "user"
#
#             contents.append({
#                 "role": role,
#                 "parts": [
#                     {
#                         "text": message["content"]
#                     }
#                 ]
#             })
#
#
#         response = client.models.generate_content(
#             model="gemini-3.6-flash",
#             contents=contents
#         )
#
#
#         answer = response.text
#
#
#         st.session_state.messages.append({
#             "role": "assistant",
#             "content": answer
#         })
#
#
#         with st.chat_message("assistant"):
#             st.write(answer)