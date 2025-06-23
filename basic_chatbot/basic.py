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

# response =graph.invoke({"messages" : "Hi"})
# print(response['messages'][-1].content)


# WITH TOOLS--------------------------------------------------------------------

from langchain_tavily import TavilySearch

search_tool = TavilySearch(max_results = 2)

# Custom tool
def multiply_tool(a: int, b:int)->int : 
    """
    Multiplies a and b

    a: first integer
    b: second integer

    returns : output integer
    """
    return a*b

tools = [search_tool,multiply_tool]
llm_with_tools = llm.bind_tools(tools)

def tool_calling_llm(state:State) : 
    return {"messages" : llm_with_tools.invoke(state["messages"])}
# Building graph ----------------------------------------------------------
from langgraph.prebuilt import ToolNode, tools_condition
builder_with_tool = StateGraph(State)
builder_with_tool.add_node("tool_calling_llm",tool_calling_llm)
builder_with_tool.add_node("tools",ToolNode(tools))

builder_with_tool.add_edge(START,"tool_calling_llm")
builder_with_tool.add_conditional_edges("tool_calling_llm", 
        # If latest message is from assistant with tool call, then --> tool_condition routes to tools node
        # ELSE : If no tool call, then tool_condition routes to END
                                        tools_condition)

builder_with_tool.add_edge("tools",END)

graph_with_tool = builder_with_tool.compile()

response = graph_with_tool.invoke({"messages" : "Give me the recent news that happened in Pimpri Chinchwad"})
for m in response['messages'] : 
    m.pretty_print()