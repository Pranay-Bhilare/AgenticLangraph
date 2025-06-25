from langgraph.types import Command
from typing import TypedDict,Annotated
from langgraph.graph import START,END,StateGraph
import operator

class State(TypedDict) : 
    text : Annotated[str,operator.add]

def node_a(state: State) :
    print("Node A")
    return Command(
        goto="node_b",
        update={"text" : "a"}
    )

def node_b(state: State): 
    print("Node B")
    return Command(
        goto="node_c", 
        update={
            "text": "b"
        }
    )

def node_c(state: State): 
    print("Node C")
    return Command(
        goto=END, 
        update={
            "text": "c"
        }
    )

graph_builder = StateGraph(State) 

graph_builder.add_node("node_a", node_a)
graph_builder.add_node("node_b", node_b)
graph_builder.add_node("node_c", node_c)

graph_builder.add_edge(START,"node_a")

graph = graph_builder.compile()
print(graph.invoke({"text" : ""}))