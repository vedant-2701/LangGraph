# LangGraph Learning Project

A comprehensive learning repository for building AI agents using LangGraph, LangChain, and Google's Gemini AI. This project contains practical examples, exercises, and fully functional agent implementations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Agents](#agents)
- [Notebooks](#notebooks)
- [Exercises](#exercises)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project demonstrates various implementations of AI agents using LangGraph, from basic chatbots to advanced RAG (Retrieval-Augmented Generation) systems. It's designed as a learning resource for understanding state graphs, agent workflows, and LLM integrations.

## ✨ Features

- **Multiple Agent Types**: Basic bots, memory-enabled agents, ReAct patterns, and RAG agents
- **Interactive Notebooks**: Jupyter notebooks for hands-on learning
- **State Management**: Examples of complex state handling with LangGraph
- **Vector Storage**: ChromaDB integration for RAG implementations
- **PDF Processing**: Document loading and chunking for knowledge retrieval
- **Google Gemini Integration**: Uses Google's Generative AI models

## 📁 Project Structure

```
LangGraph/
├── Agents/                     # Production-ready agent implementations
│   ├── 01-Agent_Bot.py        # Basic chatbot agent
│   ├── 02-Memory_Agent.py     # Agent with conversation memory
│   ├── 03-ReAct.py            # ReAct (Reasoning + Acting) pattern
│   ├── 04-Drafter.py          # Document drafting agent
│   └── 05-RAG_Agent.py        # RAG agent with vector store
├── notebooks/                  # Educational Jupyter notebooks
│   ├── Hello_World_Agent.ipynb
│   ├── Sequential_Agents.ipynb
│   ├── Conditional_Agent.ipynb
│   ├── Looping_Agent.ipynb
│   └── Multiple_Inputs.ipynb
├── exercises/                  # Practice exercises
│   ├── Exercise1.ipynb
│   ├── Exercise2.ipynb
│   ├── Exercise3.ipynb
│   ├── Exercise4.ipynb
│   └── Exercise5.ipynb
├── chroma_db_rag/             # ChromaDB vector database storage
├── logs/                       # Conversation logs and memory
├── pdfs/                       # PDF documents for RAG
├── images/                     # Project images and diagrams
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
└── .gitignore                 # Git ignore configuration
```

## 🔧 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Google API key for Gemini AI
- Git (for cloning the repository)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vedant-2701/LangGraph.git
   cd LangGraph
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

1. **Create a `.env` file** in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

2. **Get your Google API key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy it to your `.env` file

## 💻 Usage

### Running Agents

Navigate to the Agents directory and run any agent:

```bash
cd Agents
python 01-Agent_Bot.py
```

**Example interaction:**
```
Enter your message: What is artificial intelligence?
LLM Response: Artificial intelligence (AI) is...
Enter your message (or type exit/quit): 
```

### Running Notebooks

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navigate to `notebooks/` or `exercises/`
3. Open any `.ipynb` file
4. Run cells sequentially with `Shift + Enter`

### Running the RAG Agent

1. Place your PDF documents in the `pdfs/` directory
2. Run the RAG agent:
   ```bash
   cd Agents
   python 05-RAG_Agent.py
   ```

## 🤖 Agents

### 01-Agent_Bot.py
Basic chatbot that processes user messages using LangGraph state management.

**Key Features:**
- Simple state graph with START → process → END
- Direct LLM invocation
- Interactive command-line interface

### 02-Memory_Agent.py
Enhanced agent with conversation history persistence.

**Key Features:**
- Maintains conversation context
- Memory storage in logs directory
- Session-based interactions

### 03-ReAct.py
Implements the ReAct (Reasoning + Acting) pattern.

**Key Features:**
- Thought-action-observation loop
- Tool integration
- Reasoning transparency

### 04-Drafter.py
Specialized agent for document drafting and editing.

**Key Features:**
- Content generation
- Iterative refinement
- Structured output

### 05-RAG_Agent.py
Advanced Retrieval-Augmented Generation agent.

**Key Features:**
- PDF document processing
- Vector store with ChromaDB
- Semantic search and retrieval
- Context-aware responses

**How it works:**
1. Loads PDF documents
2. Chunks text using RecursiveCharacterTextSplitter
3. Creates embeddings with Google's embedding model
4. Stores vectors in ChromaDB
5. Retrieves relevant context for queries
6. Generates responses using retrieved information

## 📓 Notebooks

### Learning Path

1. **Hello_World_Agent.ipynb** - Introduction to LangGraph basics
2. **Sequential_Agents.ipynb** - Chaining multiple nodes
3. **Conditional_Agent.ipynb** - Branching logic and decisions
4. **Looping_Agent.ipynb** - Iterative agent behaviors
5. **Multiple_Inputs.ipynb** - Handling complex state

## 📝 Exercises

Practice exercises to reinforce learning:
- **Exercise1-5.ipynb**: Progressive challenges covering core concepts

## 🛠️ Technologies Used

- **[LangGraph](https://github.com/langchain-ai/langgraph)**: Framework for building stateful agents
- **[LangChain](https://python.langchain.com/)**: LLM application framework
- **[Google Gemini](https://ai.google.dev/)**: Large language model
- **[ChromaDB](https://www.trychroma.com/)**: Vector database for embeddings
- **[PyPDF](https://pypdf.readthedocs.io/)**: PDF processing
- **Python**: Core programming language

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


**Happy Learning! 🚀**
