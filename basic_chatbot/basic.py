from dotenv import load_dotenv
from langgraph.graph import START,END,StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
class State(TypedDict) :
    messages : Annotated[list,add_messages]

graph_builder = StateGraph(State)
llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

def chatbot(state:State) : 
    return {"messages" : [llm.invoke(state["messages"])]}
    
graph_builder.add_node("LLM_CHAT_BOT", chatbot)
graph_builder.add_edge(START,"LLM_CHAT_BOT")
graph_builder.add_edge("LLM_CHAT_BOT",END)

graph = graph_builder.compile()

response =graph.invoke({"messages" : "Hi"})
print(response['messages'][-1].content)

