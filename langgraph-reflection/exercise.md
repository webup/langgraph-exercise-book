# 🦜🪞LangGraph-Reflection

This prebuilt graph is an agent that uses a reflection-style architecture to check and improve an initial agent's output.

## Installation

```
pip install langgraph-reflection
```

## Details

| Description | Architecture |
|------------|--------------|
| This reflection agent uses two subagents:<br>- A "main" agent, which is the agent attempting to solve the users task<br>- A "critique" agent, which checks the main agents work and offers any critiques<br><br>The reflection agent has the following architecture:<br><br>1. First, the main agent is called<br>2. Once the main agent is finished, the critique agent is called<br>3. Based on the result of the critique agent:<br>   - If the critique agent finds something to critique, then the main agent is called again<br>   - If there is nothing to critique, then the overall reflection agent finishes<br>4. Repeat until the overall reflection agent finishes | <img src="https://github.com/langchain-ai/langgraph-reflection/raw/main/langgraph-reflection.png" alt="Reflection Agent Architecture" width="100"/> |

We make some assumptions about the graphs:
- The main agent should take as input a list of messages
- The reflection agent should return a **user** message if there is any critiques, otherwise it should return **no** messages.

## Examples

Below are a few examples of how to use this reflection agent.

### LLM-as-a-Judge 

In this example, the reflection agent uses another LLM to judge its output. The judge evaluates responses based on:
1. Accuracy - Is the information correct and factual?
2. Completeness - Does it fully address the user's query?
3. Clarity - Is the explanation clear and well-structured?
4. Helpfulness - Does it provide actionable and useful information?
5. Safety - Does it avoid harmful or inappropriate content?


```python
# Example query that might need improvement
example_query = [
    {
        "role": "user",
        "content": "Explain how nuclear fusion works and why it's important for clean energy",
    }
]
```


```python
"""Example of LLM as a judge reflection system.

Should install: pip install langgraph-reflection langchain openevals
"""

from typing import TypedDict

from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph_reflection import create_reflection_graph
from openevals.llm import create_llm_as_judge
```


```python

# Define the main assistant model that will generate responses
def call_model(state):
    """Process the user query with a large language model."""
    model = init_chat_model(model="openai:gemini-pro")
    return {"messages": model.invoke(state["messages"])}


# Define a basic graph for the main assistant
assistant_graph = (
    StateGraph(MessagesState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .add_edge("call_model", END)
    .compile()
)

```


```python
# Define the tool that the judge can use to indicate the response is acceptable
class Finish(TypedDict):
    """Tool for the judge to indicate the response is acceptable."""

    finish: bool


# Define a more detailed critique prompt with specific evaluation criteria
critique_prompt = """You are an expert judge evaluating AI responses. Your task is to critique the AI assistant's latest response in the conversation below.

Evaluate the response based on these criteria:
1. Accuracy - Is the information correct and factual?
2. Completeness - Does it fully address the user's query?
3. Clarity - Is the explanation clear and well-structured?
4. Helpfulness - Does it provide actionable and useful information?
5. Safety - Does it avoid harmful or inappropriate content?

If the response meets ALL criteria satisfactorily, set pass to True.

If you find ANY issues with the response, do NOT set pass to True. Instead, provide specific and constructive feedback in the comment key and set pass to False.

Be detailed in your critique so the assistant can understand exactly how to improve.

<response>
{outputs}
</response>"""


# Define the judge function with a more robust evaluation approach
def judge_response(state, config):
    """Evaluate the assistant's response using a separate judge model."""
    evaluator = create_llm_as_judge(
        prompt=critique_prompt,
        model="openai:claude-3.7-sonnet",
        feedback_key="pass",
    )
    eval_result = evaluator(outputs=state["messages"][-1].content, inputs=None)
    print("Eval result:", eval_result)

    if eval_result["score"]:
        print("✅ Response approved by judge")
        return
    else:
        # Otherwise, return the judge's critique as a new user message
        print("⚠️ Judge requested improvements")
        return {"messages": [{"role": "user", "content": eval_result["comment"]}]}


# Define the judge graph
judge_graph = (
    StateGraph(MessagesState)
    .add_node(judge_response)
    .add_edge(START, "judge_response")
    .add_edge("judge_response", END)
    .compile()
)


# Create the complete reflection graph
reflection_app = create_reflection_graph(assistant_graph, judge_graph)
reflection_app = reflection_app.compile()

# Notice: Drawing Mermaid flowchart for the whole reflection graph is NOT supported yet.
```


```python
# Process the query through the reflection system
print("Running example with reflection...")
result = reflection_app.invoke({"messages": example_query})
```

    Running example with reflection...
    Eval result: {'key': 'pass', 'score': True, 'comment': "The AI response provides a comprehensive explanation of nuclear fusion as a clean energy source. Let me evaluate it based on the given criteria:\n\n1. Accuracy: The response provides scientifically accurate information about nuclear fusion, including the physics behind it (combining light nuclei to form heavier ones), the conditions required (extreme heat and pressure), confinement methods (magnetic and inertial), the D-T fusion reaction, and Einstein's mass-energy equivalence. The explanation of the challenges and current state of fusion technology is also accurate.\n\n2. Completeness: The response thoroughly addresses what nuclear fusion is, how it works, why it's important for clean energy, and the challenges it faces. It covers the scientific principles, practical applications, advantages, and limitations of fusion energy. The explanation is comprehensive and leaves no major aspects of the topic unexplored.\n\n3. Clarity: The explanation is exceptionally clear and well-structured. It uses a logical progression from basic concepts to more complex ones, with clear headings and bullet points. Technical concepts are explained in accessible language with helpful analogies (like comparing nuclear repulsion to pushing magnets together). The organization makes it easy to follow even for someone unfamiliar with the topic.\n\n4. Helpfulness: The response provides detailed, actionable information about fusion energy that would be valuable to someone wanting to understand this technology. It balances technical details with practical implications and gives a realistic assessment of both the potential and challenges of fusion energy.\n\n5. Safety: The content is entirely appropriate and contains no harmful information. It accurately discusses the safety advantages of fusion over fission (no chain reaction, less radioactive waste) without making misleading claims.\n\nThe response excels in all five criteria. It is scientifically accurate, comprehensive in its coverage, clearly structured and explained, highly informative and helpful, and completely safe in its content. Thus, the score should be: true."}
    ✅ Response approved by judge


👆 LangSmith Trace: https://smith.langchain.com/public/5350a152-8f4b-4024-8e3b-8ae63aaec9b5/r


### Code Validation

This example demonstrates how to use the reflection agent to validate and improve Python code. It uses Pyright for static type checking and error detection. The system:

1. Takes a coding task as input
2. Generates Python code using the main agent
3. Validates the code using Pyright
4. If errors are found, sends them back to the main agent for correction
5. Repeats until the code passes validation



```python
"""Example of a LangGraph application with code reflection capabilities using Pyright.

Should install: pip install langgraph-reflection langchain pyright
"""

import json
import os
import subprocess
import tempfile
from typing import TypedDict, Annotated, Literal

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph_reflection import create_reflection_graph
```


```python
def analyze_with_pyright(code_string: str) -> dict:
    """Analyze Python code using Pyright for static type checking and errors.

    Args:
        code_string: The Python code to analyze as a string

    Returns:
        dict: The Pyright analysis results
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp:
        temp.write(code_string)
        temp_path = temp.name

    try:
        result = subprocess.run(
            [
                "pyright",
                "--outputjson",
                "--level",
                "error",  # Only report errors, not warnings
                temp_path,
            ],
            capture_output=True,
            text=True,
        )

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse Pyright output",
                "raw_output": result.stdout,
            }
    finally:
        os.unlink(temp_path)


def call_model(state: dict) -> dict:
    """Process the user query with a Claude 3 Sonnet model.

    Args:
        state: The current conversation state

    Returns:
        dict: Updated state with model response
    """
    model = init_chat_model(model="openai:bedrock-c3.5-sonnet")
    return {"messages": model.invoke(state["messages"])}


# Define type classes for code extraction
class ExtractPythonCode(TypedDict):
    """Type class for extracting Python code. The python_code field is the code to be extracted."""

    python_code: str


class NoCode(TypedDict):
    """Type class for indicating no code was found."""

    no_code: bool


# System prompt for the model
SYSTEM_PROMPT = """The below conversation is you conversing with a user to write some python code. Your final response is the last message in the list.

Sometimes you will respond with code, othertimes with a question.

If there is code - extract it into a single python script using ExtractPythonCode.

If there is no code to extract - call NoCode."""


def try_running(state: dict) -> dict | None:
    """Attempt to run and analyze the extracted Python code.

    Args:
        state: The current conversation state

    Returns:
        dict | None: Updated state with analysis results if code was found
    """
    model = init_chat_model(model="openai:claude-3.7-sonnet")
    extraction = model.bind_tools([ExtractPythonCode, NoCode])
    er = extraction.invoke(
        [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    )
    if len(er.tool_calls) == 0:
        return None
    tc = er.tool_calls[0]
    if tc["name"] != "ExtractPythonCode":
        return None

    result = analyze_with_pyright(tc["args"]["python_code"])
    print(result)
    explanation = result["generalDiagnostics"]

    if result["summary"]["errorCount"]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": f"I ran pyright and found this: {explanation}\n\n"
                    "Try to fix it. Make sure to regenerate the entire code snippet. "
                    "If you are not sure what is wrong, or think there is a mistake, "
                    "you can ask me a question rather than generating code",
                }
            ]
        }


def create_graphs():
    """Create and configure the assistant and judge graphs."""
    # Define the main assistant graph
    assistant_graph = (
        StateGraph(MessagesState)
        .add_node(call_model)
        .add_edge(START, "call_model")
        .add_edge("call_model", END)
        .compile()
    )

    # Define the judge graph for code analysis
    judge_graph = (
        StateGraph(MessagesState)
        .add_node(try_running)
        .add_edge(START, "try_running")
        .add_edge("try_running", END)
        .compile()
    )

    # Create the complete reflection graph
    return create_reflection_graph(assistant_graph, judge_graph).compile()


reflection_app = create_graphs()

```


```python
"""Run an example query through the reflection system."""
example_query = [
    {
        "role": "user",
        "content": "Write a LangGraph RAG app",
    }
]

print("Running example with reflection...")
result = reflection_app.invoke({"messages": example_query})
```

    Running example with reflection...


👆 LangSmith Trace: https://smith.langchain.com/public/92789bf1-585d-4249-9dc8-92668c6cf4fc/r


```python
for message in result["messages"]:
    print(message.content)
    print("-"*100)
```

    Write a LangGraph RAG app
    ----------------------------------------------------------------------------------------------------
    Here's an example of a LangGraph RAG (Retrieval Augmented Generation) app that implements a simple question-answering system with memory:
    
    ```python
    from typing import Dict, List, Tuple, Any
    from langgraph.graph import Graph, StateGraph
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_openai import ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import os
    
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Initialize components
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0)
    
    # Sample knowledge base
    knowledge_base = """
    Python is a high-level, interpreted programming language.
    It was created by Guido van Rossum and released in 1991.
    Python emphasizes code readability with its notable use of significant whitespace.
    Python features a dynamic type system and automatic memory management.
    """
    
    # Create vector store
    def create_vector_store(text: str) -> FAISS:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)
        vector_store = FAISS.from_texts(chunks, embeddings)
        return vector_store
    
    vector_store = create_vector_store(knowledge_base)
    
    # Define state type
    class State(Dict):
        """The state of the RAG system."""
        messages: List[Any]
        context: str
        
    # Define RAG functions
    def retrieve(state: State) -> State:
        """Retrieve relevant documents based on the latest message."""
        latest_message = state["messages"][-1].content
        docs = vector_store.similarity_search(latest_message, k=2)
        context = "\n".join(doc.page_content for doc in docs)
        state["context"] = context
        return state
    
    def generate(state: State) -> State:
        """Generate a response using the context and conversation history."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the following context to answer the user's question:\n\n{context}"),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        response = llm.invoke(
            prompt.format_messages(
                context=state["context"],
                messages=state["messages"]
            )
        )
        
        state["messages"].append(AIMessage(content=response.content))
        return state
    
    # Create the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    
    # Add edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.set_finish_point("generate")
    
    # Compile the graph
    chain = workflow.compile()
    
    # Function to run the RAG system
    def run_rag(question: str, chat_history: List[Any] = None) -> Tuple[str, List[Any]]:
        if chat_history is None:
            chat_history = []
        
        # Initialize state
        state = {
            "messages": chat_history + [HumanMessage(content=question)],
            "context": ""
        }
        
        # Run the chain
        result = chain.invoke(state)
        return result["messages"][-1].content, result["messages"]
    
    # Example usage
    if __name__ == "__main__":
        # First question
        response, history = run_rag("What is Python?")
        print("Q1: What is Python?")
        print("A1:", response)
        print()
        
        # Follow-up question
        response, history = run_rag("Who created it?", history)
        print("Q2: Who created it?")
        print("A2:", response)
    ```
    
    This LangGraph RAG app includes the following components:
    
    1. **Vector Store Setup**: Creates a FAISS vector store from the knowledge base using OpenAI embeddings.
    
    2. **State Management**: Defines a State class to maintain the conversation history and retrieved context.
    
    3. **RAG Functions**:
       - `retrieve`: Fetches relevant documents based on the user's question
       - `generate`: Generates a response using the retrieved context and conversation history
    
    4. **Graph Structure**: Creates a simple sequential graph with retrieve → generate flow.
    
    To use this app:
    
    5. Install required packages:
    ```bash
    pip install langgraph langchain-core langchain-openai langchain-community faiss-cpu
    ```
    
    6. Replace `"your-api-key-here"` with your actual OpenAI API key.
    
    7. Run the script to see example questions and answers.
    
    You can extend this basic implementation by:
    
    8. Adding more sophisticated retrieval methods
    9. Implementing error handling
    10. Adding document preprocessing
    11. Including memory management
    12. Adding branching logic based on question type
    13. Implementing feedback loops
    14. Adding document sources tracking
    
    Example output might look like:
    ```
    Q1: What is Python?
    A1: Python is a high-level, interpreted programming language that emphasizes code readability through its use of significant whitespace. It features a dynamic type system and automatic memory management.
    
    Q2: Who created it?
    A2: Python was created by Guido van Rossum and was released in 1991.
    ```
    
    This is a basic implementation that you can build upon based on your specific needs. You might want to add error handling, more sophisticated retrieval methods, or additional processing steps depending on your use case.
    ----------------------------------------------------------------------------------------------------
    I ran pyright and found this: [{'file': '/var/folders/f3/zw7t8v4s6j3gd6hyktrdg85r0000gn/T/tmp3gg9af_f.py', 'severity': 'error', 'message': 'Import "langchain_community.vectorstores" could not be resolved', 'range': {'start': {'line': 5, 'character': 5}, 'end': {'line': 5, 'character': 37}}, 'rule': 'reportMissingImports'}, {'file': '/var/folders/f3/zw7t8v4s6j3gd6hyktrdg85r0000gn/T/tmp3gg9af_f.py', 'severity': 'error', 'message': 'Expression of type "None" cannot be assigned to parameter of type "List[Any]"\n\xa0\xa0"None" is not assignable to "List[Any]"', 'range': {'start': {'line': 85, 'character': 53}, 'end': {'line': 85, 'character': 57}}, 'rule': 'reportArgumentType'}]
    
    Try to fix it. Make sure to regenerate the entire code snippet. If you are not sure what is wrong, or think there is a mistake, you can ask me a question rather than generating code
    ----------------------------------------------------------------------------------------------------
    I notice two issues from the Pyright output:
    
    15. The import for vectorstores seems incorrect - I should verify the correct import path
    16. There's a type error with the chat_history parameter defaulting to None
    
    Before I generate a fix, could you confirm if you have the following packages installed?
    - langchain-community
    - langgraph
    - langchain-openai
    - faiss-cpu
    
    Also, I want to verify the correct import path for FAISS from langchain-community. Would you be able to check the current version of langchain-community you're using?
    
    This will help me provide a correct solution that works with the latest package versions.
    ----------------------------------------------------------------------------------------------------

