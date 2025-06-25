from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command, interrupt
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import uuid

llm = ChatGoogleGenerativeAI(model = "gemini-flash-2.0")
class State(TypedDict) : 
    linkedin_topic: str
    generated_post: Annotated[List[str], add_messages]
    human_feedback: Annotated[List[str], add_messages]

def model(state:State): 
    print("[model] Generating content")
    linkedin_topic = state["linkedin_topic"]
    feedback = state["human_feedback"] if "human_feedback" in state else ["No Feedback yet"]

    prompt = f"""
        LinkedIn Topic: {linkedin_topic}
        Human Feedback: {feedback[-1] if feedback else "No feedback yet"}

        Generate a structured and well-written LinkedIn post based on the given topic.

        Consider previous human feedback to refine the reponse. 
    """

    response = llm.invoke([
        SystemMessage(content="You are an expert LinkedIn content writer"), 
        HumanMessage(content=prompt)
    ])

    geneated_linkedin_post = response.content

    print(f"[model_node] Generated post:\n{geneated_linkedin_post}\n")

    return {
       "generated_post": [AIMessage(content=geneated_linkedin_post)] , 
       "human_feedback": feedback
    }

def human_node(state: State): 
    print("\n [human_node] awaiting human feedback...")

    generated_post = state["generated_post"]

    user_feedback = interrupt(
        {
            "generated_post": generated_post, 
            "message": "Provide feedback or type 'done' to finish"
        }
    )

    print(f"[human_node] Received human feedback: {user_feedback}")

    if user_feedback.lower() == "done": 
        return Command(update={"human_feedback": state["human_feedback"] + ["Finalised"]}, goto="end_node")

    return Command(update={"human_feedback": state["human_feedback"] + [user_feedback]}, goto="model")

def end_node(state: State): 
    print("\n[end_node] Process finished")
    print("Final Generated Post:", state["generated_post"][-1])
    print("Final Human Feedback", state["human_feedback"])
    return {"generated_post": state["generated_post"], "human_feedback": state["human_feedback"]}


graph = StateGraph(State)
graph.add_node("model", model)
graph.add_node("human_node", human_node)
graph.add_node("end_node", end_node)

graph.add_edge(START, "model")
graph.add_edge("model", "human_node")

graph.set_finish_point("end_node")

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

thread_config = {"configurable": {
    "thread_id": uuid.uuid4()
}}

linkedin_topic = input("Enter your LinkedIn topic: ")

initial_state = {
    "linkedin_topic": linkedin_topic, 
    "generated_post": [], 
    "human_feedback": []
}

for chunk in app.stream(initial_state, config=thread_config):
    for node_id, value in chunk.items():

        if(node_id == "__interrupt__"):
            while True: 
                user_feedback = input("Provide feedback (or type 'done' when finished): ")
                app.invoke(Command(resume=user_feedback), config=thread_config)
                if user_feedback.lower() == "done":
                    break