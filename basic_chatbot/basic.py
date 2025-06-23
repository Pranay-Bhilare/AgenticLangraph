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

search_tool = TavilySearch()

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

# response = graph_with_tool.invoke({"messages" : "Give me the recent news that happened in Pimpri Chinchwad"})
# for m in response['messages'] : 
#     m.pretty_print()



# ------------ ReAct Style---------------------
builder_react = StateGraph(State)
builder_react.add_node("tool_calling_llm",tool_calling_llm)
builder_react.add_node("tools", ToolNode(tools))

builder_react.add_edge(START,"tool_calling_llm")
builder_react.add_edge("tools","tool_calling_llm")
builder_react.add_conditional_edges("tool_calling_llm",tools_condition)

graph_react = builder_react.compile()

# response = graph_react.invoke({"messages" : "Multiply 2 with 3 and then give me that number of recent news that happened in Pimpri Chinchwad, tell me all the news, and then again multiply 1 with 5 , and give me that number of AI news related to India, and then give me all final results"})
# for m in response['messages'] : 
#     m.pretty_print()



# ---------------------- Adding memory ------------------------------- 

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

builder_react_with_memory = StateGraph(State)
builder_react_with_memory.add_node("tool_calling_llm",tool_calling_llm)
builder_react_with_memory.add_node("tools", ToolNode(tools))

builder_react_with_memory.add_edge(START,"tool_calling_llm")
builder_react_with_memory.add_edge("tools","tool_calling_llm")
builder_react_with_memory.add_conditional_edges("tool_calling_llm",tools_condition)

graph_react_with_memory = builder_react_with_memory.compile(checkpointer=memory)
config = {"configurable" : {"thread_id":"1"}}

response = graph_react_with_memory.invoke({"messages" : "Hi, My name is Pranay Bhilare"},config=config)
print(response['messages'][-1].content)
response = graph_react_with_memory.invoke({"messages" : "Tell me the latest news of what happened in Pune, I need to know 3 news"}, config=config)
print(response['messages'][-1].content)
# CHECKING THE MEMORY, IT SHOULD REMEBER BY NAME
response = graph_react_with_memory.invoke({"messages" : "Hey do you remember my name ?"}, config=config)
print(response['messages'][-1].content)


