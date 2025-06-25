from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.messages import HumanMessage
from langchain_community.tools import TavilySearchResults
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

parser = PydanticOutputParser(pydantic_object=AnswerQuestion)

def tavily_search(query: str) -> str:
    """Search the web using Tavily for the given query."""
    tavily = TavilySearchResults(max_results=5)
    return tavily.invoke(query)

tools_responder = [tavily_search, AnswerQuestion]
tools_revisor = [tavily_search, ReviseAnswer]

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AI researcher.
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time = lambda: datetime.datetime.now().isoformat)

def pydantic_validator(response) : 
    pydantic_object = AnswerQuestion.model_validate(response.tool_calls[0]["args"])
    return pydantic_object
first_responder_prompt_template = actor_prompt_template.partial(first_instruction = "Provide a detailed ~250 word answer")

# parsing = RunnableLambda(pydantic_validator)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice = "AnswerQuestion")

# response = first_responder_chain.invoke({"messages" : [HumanMessage(content="Write me a blog post on how developers can leverage AI for growth")]})

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""
revisor_chain = actor_prompt_template.partial(first_instruction = revise_instructions) | llm.bind_tools(tools=[ReviseAnswer], tool_choice = "ReviseAnswer")

