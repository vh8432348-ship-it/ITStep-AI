from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import dotenv
import os

dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")



llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=0.1,
    max_output_tokens=1000
)


class NameBooks(BaseModel):
    genre: str = Field(description="Жанр книги")


parser = PydanticOutputParser(pydantic_object=NameBooks)

instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Ти визначаєш жанр книги.

        {instructions}
        """
    ),
    (
        "human",
        "Визнач жанр книги: {book}"
    )
])

chain = prompt | llm | parser


result = chain.invoke({
    "book": "1984",
    "instructions": instructions
})

print(result)
print(result.genre)

class RecommendedBooks(BaseModel):
    books: list[str] = Field(
        description="Список схожих книг"
    )

parser_recommendations = PydanticOutputParser(
    pydantic_object=RecommendedBooks
)

instructions_recommendations = (
    parser_recommendations.get_format_instructions()
)

prompt_recommendations = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Ти рекомендуєш книги.

        {instructions}
        """
    ),
    (
        "human",
        """
        Назва книги: {book}
        Жанр книги: {genre}

        Підбери 5 схожих книг.
        Частина повинна бути того самого жанру,
        частина може бути іншого жанру,
        але схожа за сюжетом, тематикою або атмосферою.
        """
    )
])