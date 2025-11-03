from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Use any one of the two whichever works for you
# from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

def process(state: AgentState) -> AgentState:
    """This node processes the conversation by sending messages to the LLM and appending the response."""
    # Make a call to the LLM with the current messages
    response = llm.invoke(state["messages"])

    # Append the LLM's response to the messages
    state["messages"].append(AIMessage(content=response.content))

    print("\nLLM Response:", response.content)

    print("CURRENT STATE: ", state["messages"])

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

conversation_history: List[Union[HumanMessage, AIMessage]] = []

user_input = input("Enter your message: ")

while user_input.lower() not in ["exit", "quit"]:
    conversation_history.append(HumanMessage(content=user_input))

    # Invoke the agent with the current conversation history
    result = agent.invoke({"messages": conversation_history})

    # print(result["messages"])

    conversation_history = result["messages"]

    user_input = input("Enter your message (or type exit/quit): ")

# Save the conversation history to a file
# Not a good practice for sensitive data, just for demonstration
with open("../logs/conversation_memory.txt", "w", encoding="utf-8") as f:
    
    f.write("=" * 60 + "\n")
    f.write("Conversation History:\n\n")
    f.write("=" * 60 + "\n")

    for message in conversation_history:
        role = "Human" if isinstance(message, HumanMessage) else "AI"
        f.write(f"{role}: {message.content}\n")

        if role == "AI":
            f.write("-" * 60 + "\n")  # Separator after AI messages
    f.write("\n" + "=" * 60 + "\n")
    f.write("\nEnd of Conversation\n")
    f.write("=" * 60 + "\n")

print("Conversation history saved to conversation_memory.txt")