import json
from typing import List, TypedDict, Any, Annotated
from langchain_core.messages import AIMessage, ToolMessage
from langchain_community.tools import TavilySearchResults
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

tavily_tool = TavilySearch(max_results = 5)

class State(TypedDict) :
    messages : Annotated[list,add_messages]

def execute_tool(state: State) :
    print("[TRACE] Entered execute_tool", flush=True)
    last_ai_message: AIMessage = state["messages"][-1]
    print(f"[TRACE] last_ai_message: {last_ai_message}", flush=True)
    print("--- Executing tools ---", flush=True)
    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        print("[TRACE] No tool_calls found in last_ai_message", flush=True)
        return {"messages": state["messages"]}
    
    tool_messages = []
    print(f"[TRACE] tool_calls: {last_ai_message.tool_calls}", flush=True)
    
    for tool_call in last_ai_message.tool_calls:
        print(f"[TRACE] Processing tool_call: {tool_call}", flush=True)
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            print(f"[TRACE] call_id: {call_id}", flush=True)
            search_queries = tool_call["args"].get("search_queries", [])
            print(f"[TRACE] search_queries: {search_queries}", flush=True)
            
            query_results = {}
            for query in search_queries:
                print(f"[TRACE] Invoking tavily_tool with query: {query}", flush=True)
                try:
                    result = tavily_tool.invoke(query)
                    print(f"[TRACE] Result for query '{query}': {result}", flush=True)
                except Exception as e:
                    print(f"[ERROR] Exception during tavily_tool.invoke: {e}", flush=True)
                    result = f"[ERROR] {e}"
                query_results[query] = result
            
            tool_message = ToolMessage(
                content=json.dumps(query_results),
                tool_call_id=call_id
            )
            print(f"[TRACE] Appending tool_message: {tool_message}", flush=True)
            tool_messages.append(tool_message)
    print("Done executing tools, returning the tool message", flush=True)
    print(f"[TRACE] Returning tool_messages: {tool_messages}", flush=True)
    return {"messages" : tool_messages}

    

    