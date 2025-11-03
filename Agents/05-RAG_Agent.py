import os
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from operator import add as add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.tools import tool
from dotenv import load_dotenv
# from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings

load_dotenv()

# temperature=0 for more deterministic results
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)   

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

pdf_path = "../pdfs/Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

# Load the pdf document
pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"Loaded {len(pages)} pages from the PDF document.")
except Exception as e:
    print(f"Error loading PDF document: {e}")
    raise RuntimeError(f"Error loading PDF document: {e}")

# Chunk the document into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # size of each chunk
    chunk_overlap=200   # overlap between chunks (how much context to keep in both adjacent chunks)
)

# Split the pages into chunks
pages_split = text_splitter.split_documents(pages)

persist_directory = "../chroma_db_rag"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# Create Chroma vector store using embedding model 
try:
    vector_store = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    # vector_store.persist()
    print(f"Chroma vector store created and persisted at {persist_directory}.")
except Exception as e:
    print(f"Error creating Chroma vector store: {e}")
    raise RuntimeError(f"Error creating Chroma vector store: {e}")

# Retriever to fetch relevant documents
retriever = vector_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 5}  # number of similar documents to retrieve
)

@tool
def retriever_tool(query: str) -> str:
    """Retrieve relevant document chunks based on the query."""
    relevant_docs = retriever.invoke(query)

    if not relevant_docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."
    
    combined_content = "\n\n".join([f"Document {i + 1}: \n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
    return combined_content

tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> bool:
    """Checks if the agent should continue or stop based on the last message."""
    last_message = state["messages"][-1]
    
    return hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

# Creating a dictionary for our tools for easy access
tools_dict = {
    our_tool.name: our_tool for our_tool in tools
}


# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Calls the LLM with the current state and returns the updated state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}

# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Takes action based on the last message's tool calls."""
    tool_calls = state['messages'][-1].tool_calls

    results = []

    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present from our tools_dict
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            

        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")

    return { 'messages': results }

# Create the state graph
graph = StateGraph(AgentState)

# add nodes to the graph
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

# Add edges to the graph
graph.add_edge(START, "llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)
graph.add_edge("retriever_agent", "llm")

rag_agent = graph.compile()


def running_agent():
    print("\n=== RAG AGENT===")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


running_agent()