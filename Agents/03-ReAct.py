# ReAct - Reasoning and Acting Agent

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage # The foundational class for all message types
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool (content and tool_call_id)
from langchain_core.messages import SystemMessage # Message type for providing instructions to LLM
from langchain_core.tools import tool

# Reducer Function
# Rule that controls how updates from nodes are combined with the existing state.
# Tells us how to merge new data into the current state
# Without a reducer, updates would have replaced the existing value entirely!
from langgraph.graph.message import add_messages    # reducer function to add messages to the state

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

# Use any one of the two whichever works for you
# from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Annotated - provides additional context (metadata) w/o affecting the type
# Sequence - a generic version of list, tuple, etc.
# To automatically handle the state updates for sequences such as by adding new messages to a chat history


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts two numbers."""
    return a - b

tools = [add, multiply, subtract]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    
    if not last_message.tool_calls:
        return "end"  # No tool calls, so we are done
    else:
        return "continue"  # There are tool calls, so continue
    

graph = StateGraph(AgentState)

graph.add_node("model_call", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "model_call")
graph.add_conditional_edges(
    "model_call",
    should_continue,
    {
        "continue": "tool_node",
        "end": END,
    },
)
graph.add_edge("tool_node", "model_call")

agent = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]

        if isinstance(message, tuple):
            print("Tool Call Response:", message)
        else:
            message.pretty_print()
            

inputs = { "messages": [(
    "user",
    "Add 40 and 12, Subtract 20 and 8, multiply the results by 2. Also tell me a joke."
)]}


print_stream(agent.stream(inputs, stream_mode="values"))