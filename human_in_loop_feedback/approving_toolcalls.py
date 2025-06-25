from langchain_tavily import TavilySearch
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages
from dotenv import load_dotenv

load_dotenv()

memory = MemorySaver()
search_tool = TavilySearch()
tools = [search_tool]

class State(TypedDict):
    messages : Annotated[list,add_messages]

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools=tools)

def llm_node(state:State) :
    return {"messages" : [llm_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("llm_node", llm_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START,"llm_node")
graph_builder.add_conditional_edges("llm_node",tools_condition)
graph_builder.add_edge("tools","llm_node")

graph = graph_builder.compile(checkpointer=memory, interrupt_before=["tools"])
config = {"configurable": {
    "thread_id": 1
}}

events = graph.stream({
        "messages": [HumanMessage(content="Give me the recent news that happened in Pimpri Chinchwad")]
    }, config=config, stream_mode="values")

for event in events :
    event["messages"][-1].pretty_print()

events = graph.stream(None, config, stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()