print("[TRACE] Starting graph_reflexion.py", flush=True)
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from chains import first_responder_chain, revisor_chain
from execute_tools import execute_tool, State

print("[TRACE] Creating StateGraph", flush=True)
graph_builder = StateGraph(State)
print("[TRACE] Adding nodes", flush=True)
graph_builder.add_node("responder", first_responder_chain)
graph_builder.add_node("execute_tools", execute_tool)
graph_builder.add_node("revisor", revisor_chain)

print("[TRACE] Adding edges", flush=True)
graph_builder.add_edge(START,"responder")
graph_builder.add_edge("responder", "execute_tools")
graph_builder.add_edge("execute_tools", "revisor")

def event_loop(state: State) -> str:
    print(f"[TRACE] In event_loop with state: {state}", flush=True)
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    num_iterations = count_tool_visits
    print(f"[TRACE] num_iterations: {num_iterations}", flush=True)
    if num_iterations > 2:
        print("[TRACE] Returning END from event_loop", flush=True)
        return END
    else : 
        print("[TRACE] Returning 'execute_tools' from event_loop", flush=True)
        return "execute_tools"

print("[TRACE] Adding conditional edges", flush=True)
graph_builder.add_conditional_edges("revisor", event_loop)

print("[TRACE] Compiling graph", flush=True)
graph = graph_builder.compile()

print("[TRACE] Invoking graph", flush=True)
response = graph.invoke(
    {"messages": [HumanMessage("Write about how small business can leverage AI to grow")]}
)

print("[TRACE] Graph invoke complete", flush=True)
print(f"[TRACE] response: {response}", flush=True)
print(response[-1].tool_calls[0]["args"]["answer"])
print("\n\n\n\n")
print(response, "response")