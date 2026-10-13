import streamlit as st

from exam.agent import agent


st.set_page_config(
    page_title="Медичний центр",
    page_icon="🏥",
)


st.title("🏥 Чат-бот медичного центру")
st.write(
    "Задайте питання про документи медичного центру, "
    "лікарів, пацієнтів або записи на прийом."
)


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Введіть ваше питання..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Обробка запиту..."):

            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ]
                }
            )

            response = result["messages"][-1].content

            if isinstance(response, list):
                response_text = ""

                for item in response:
                    if item.get("type") == "text":
                        response_text += item["text"]

            else:
                response_text = response

            st.markdown(response_text)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
        }
    )