from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Use any one of the two whichever works for you
# from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define the node
def process(state: AgentState) -> AgentState:
    # Make a call to the LLM with the current messages
    response = llm.invoke(state["messages"])

    # Append the LLM's response to the messages
    state["messages"].append(HumanMessage(content=response.content))

    print("\nLLM Response:", response.content)

    # Return the updated state
    return state

# Build the state graph
graph = StateGraph(AgentState)

# Add nodes 
graph.add_node("process", process)

# Define edges
graph.add_edge(START, "process")
graph.add_edge("process", END)

# Compile the agent
agent = graph.compile()

user_input = input("Enter your message: ")

while user_input.lower() not in ["exit", "quit"]:
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter your message (or type exit/quit): ")
