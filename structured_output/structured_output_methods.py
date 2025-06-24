from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

# ------------ FIRST WAY OF STRUCTURED OUTPUT -----------------------------
class Country(BaseModel) :
    """Information about a country"""

    name : str = Field(description="name of the country")
    language : str = Field(description="language of the country")
    capital : str = Field(description="capital of the country")
    
llm_structured_output_1 = llm.with_structured_output(Country)
print(llm_structured_output_1.invoke("Tell me about Germany"))


# ------------------- SECOND WAY OF STRUCTURED OUTPUT -------------------------

from typing_extensions import TypedDict,Annotated

class Country2(TypedDict) :
    """Information about a country"""
    
    name : Annotated[str, ..., "name of the country"]
    language : Annotated[str, ..., "language of the country"]
    capital : Annotated[str, ..., "capital of the country"]

llm_structured_output_2 = llm.with_structured_output(Country2)
print(llm_structured_output_2.invoke("Tell me about Germany"))