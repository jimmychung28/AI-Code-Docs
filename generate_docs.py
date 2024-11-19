from typing import Dict, TypedDict
from langgraph.graph import Graph
import operator
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Define types for state
class AgentState(TypedDict):
    messages: list
    code: str
    documentation: str
    current_section: str

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7
)

# Create prompts for different documentation sections
code_analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code analysis expert. Analyze the given code and identify its main components, structure, and purpose."),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "Analyze this code:\n{code}")
])

docstring_generator_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a documentation expert. Generate comprehensive docstrings for the given code."),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "Generate docstrings for this code based on the analysis:\n{code}\n\nAnalysis:\n{analysis}")
])

usage_example_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical writer. Create clear usage examples for the given code."),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "Create usage examples for this code:\n{code}")
])

def analyze_code(state: AgentState) -> AgentState:
    """Analyzes the code and identifies its components and structure."""
    messages = code_analyzer_prompt.format_messages(
        messages=state["messages"],
        code=state["code"]
    )
    response = llm.invoke(messages)
    state["messages"].append(HumanMessage(content=messages[-1].content))
    state["messages"].append(response)
    state["current_section"] = "analysis"
    return state

def generate_docstrings(state: AgentState) -> AgentState:
    """Generates comprehensive docstrings based on code analysis."""
    analysis = state["messages"][-1].content
    messages = docstring_generator_prompt.format_messages(
        messages=state["messages"],
        code=state["code"],
        analysis=analysis
    )
    response = llm.invoke(messages)
    state["messages"].append(HumanMessage(content=messages[-1].content))
    state["messages"].append(response)
    state["documentation"] = response.content
    state["current_section"] = "docstrings"
    return state

def create_usage_examples(state: AgentState) -> AgentState:
    """Creates usage examples for the documented code."""
    messages = usage_example_prompt.format_messages(
        messages=state["messages"],
        code=state["code"]
    )
    response = llm.invoke(messages)
    state["messages"].append(HumanMessage(content=messages[-1].content))
    state["messages"].append(response)
    state["documentation"] += "\n\nUsage Examples:\n" + response.content
    state["current_section"] = "examples"
    return state

def should_continue(state: AgentState) -> str:
    """Determines the next step in the documentation process."""
    if state["current_section"] == "":
        return "analyze"
    elif state["current_section"] == "analysis":
        return "docstrings"
    elif state["current_section"] == "docstrings":
        return "examples"
    elif state["current_section"] == "examples":
        return "end"
    return "end"

def end_documentation(state: AgentState) -> Dict:
    """Final node that returns the state unchanged."""
    return state

# Create the workflow graph
workflow = Graph()

# Add nodes to the graph
workflow.add_node("analyze", analyze_code)
workflow.add_node("docstrings", generate_docstrings)
workflow.add_node("examples", create_usage_examples)
workflow.add_node("end", end_documentation)

# Set the entry point
workflow.set_entry_point("analyze")

# Add edges with proper end state handling
workflow.add_conditional_edges(
    "analyze",
    should_continue,
    {
        "analyze": "analyze",
        "docstrings": "docstrings",
        "examples": "examples",
        "end": "end"
    }
)

workflow.add_conditional_edges(
    "docstrings",
    should_continue,
    {
        "docstrings": "docstrings",
        "examples": "examples",
        "end": "end"
    }
)

workflow.add_conditional_edges(
    "examples",
    should_continue,
    {
        "examples": "examples",
        "end": "end"
    }
)

# Set the end node
workflow.set_finish_point("end")

# Compile the graph
app = workflow.compile()

def generate_documentation(code: str) -> str:
    """
    Generates comprehensive documentation for the given code.
    
    Args:
        code (str): The source code to document
        
    Returns:
        str: Generated documentation including docstrings and usage examples
    """
    # Initialize the state
    initial_state = {
        "messages": [],
        "code": code,
        "documentation": "",
        "current_section": ""
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    return final_state["documentation"]

# Example usage
if __name__ == "__main__":
    sample_code = """
    def fibonacci(n: int) -> int:
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """
    
    documentation = generate_documentation(sample_code)
    print("Generated Documentation:")
    print(documentation)