from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage,AIMessage
from chains import Chain
chain_class = Chain()
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def generation_node(state: State):
    chain = chain_class.generation_chain()
    response = chain.invoke({"messages": state["messages"]})
    content = response.content.strip()
    if content:
        return {"messages": [AIMessage(content=content)]}
    else:
        print("DEBUG: generation_node got empty response, skipping message.")
        return {"messages": []}

def critique_node(state: State):
    chain = chain_class.critique_chain()
    response = chain.invoke({"messages": state["messages"]})
    content = getattr(response, 'content', response).strip()
    if content:
        return {"messages": [HumanMessage(content=content)]}
    else:
        print("DEBUG: critique_node got empty response, skipping message.")
        return {"messages": []}
    

graph_builder.add_node("generation_node", generation_node)
graph_builder.add_node("critique_node", critique_node)
graph_builder.add_edge(START, "generation_node")

def should_continue(state: State):
    if len(state["messages"]) > 6:
        return END
    else:
        return "critique_node"

graph_builder.add_conditional_edges("generation_node", should_continue)
graph_builder.add_edge("critique_node", "generation_node")

graph = graph_builder.compile()
response = graph.invoke({"messages": [HumanMessage(content="AI Agents taking over content creation")]})
print("\n\n\n------------------- FINAL RESPONSE ----------------------:\n\n\n", response["messages"][-1].content)