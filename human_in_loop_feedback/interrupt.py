from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from typing import TypedDict,Annotated
from langgraph.checkpoint.memory import MemorySaver
import operator
memory = MemorySaver()

class State(TypedDict):
    value: Annotated[str,operator.add]

def node_a(state: State): 
    print("Node A")
    return Command(
        goto="node_b", 
        update={
            "value": "a"
        }
    )

def node_b(state: State): 
    print("Node B")

    human_response = interrupt("Do you want to go to C or D? Type C/D")

    print("Human Review Values: ", human_response)
    
    if(human_response == "C"): 
        return Command(
            goto="node_c", 
            update={
                "value": "b"
            }
        ) 
    elif(human_response == "D"): 
        return Command(
            goto="node_d", 
            update={
                "value": "b"
            }
        )


def node_c(state: State):
    print("Node C")
    return Command(
        goto=END, 
        update={
            "value": "c"
        }
    )

def node_d(state: State)->State: 
    print("Node D")
    return Command(
        goto=END, 
        update={
            "value": "d"
        }
)

graph = StateGraph(State)

graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)
graph.add_node("node_d", node_d)

graph.add_edge(START,"node_a") 

graph_ = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

initialState = {
    "value": ""
}
first_result = graph_.invoke(initialState, config, stream_mode="updates")
print(first_result)

print(graph_.get_state(config).next)
second_result = graph_.invoke(Command(resume="C"), config=config, stream_mode="updates")
print(second_result)
