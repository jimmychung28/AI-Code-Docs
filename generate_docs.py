from typing import Dict, TypedDict, Optional, Set, List, Union, Annotated
from dataclasses import dataclass, field
from enum import Enum
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from langgraph.graph.tools import Tool
import operator
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import json
from typing import TypeVar, Generic
from functools import partial

# Define types
class DocumentationType(str, Enum):
    ANALYSIS = "analysis"
    DOCSTRINGS = "docstrings"
    EXAMPLES = "examples"
    TESTS = "tests"
    API_SPEC = "api_spec"
    ARCHITECTURE = "architecture"
    DEPLOYMENT = "deployment"
    SECURITY = "security"
    PERFORMANCE = "performance"

class AnalysisLevel(str, Enum):
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

# State definitions
class DocumentationTask(BaseModel):
    """Represents a single documentation task in the workflow"""
    doc_type: DocumentationType
    dependencies: Set[DocumentationType] = Field(default_factory=set)
    status: str = "pending"
    content: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class WorkflowState(BaseModel):
    """Complete state of the documentation workflow"""
    tasks: Dict[DocumentationType, DocumentationTask] = Field(default_factory=dict)
    code: str
    messages: List = Field(default_factory=list)
    analysis_level: AnalysisLevel = AnalysisLevel.DETAILED
    current_tasks: Set[DocumentationType] = Field(default_factory=set)
    completed_tasks: Set[DocumentationType] = Field(default_factory=set)
    failed_tasks: Set[DocumentationType] = Field(default_factory=set)
    results: Dict[str, str] = Field(default_factory=dict)

class DocumentationTools:
    """Tools for documentation generation"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.prompts = self._initialize_prompts()
        
    def _initialize_prompts(self) -> Dict[DocumentationType, ChatPromptTemplate]:
        return {
            DocumentationType.ANALYSIS: ChatPromptTemplate.from_messages([
                ("system", """Analyze the code structure, patterns, and complexity.
                Detail Level: {analysis_level}
                
                Provide your analysis with:
                - Main components and their relationships
                - Code patterns and architectural decisions
                - Complexity assessment
                - Key dependencies
                
                Format as markdown with clear sections."""),
                ("human", "{code}")
            ]),
            
            DocumentationType.API_SPEC: ChatPromptTemplate.from_messages([
                ("system", """Generate OpenAPI 3.0 specification for the code.
                Include:
                - All endpoints and their methods
                - Request/response schemas
                - Authentication requirements
                - Error responses
                
                Base your analysis on the provided code analysis.
                Format as valid OpenAPI JSON."""),
                ("human", "Code: {code}\nAnalysis: {analysis}")
            ]),
            
            # Add other specialized prompts...
        }
    
    def create_tools(self) -> List[Tool]:
        """Create tools for the workflow"""
        return [
            Tool(
                name=doc_type.value,
                description=f"Generate {doc_type.value} documentation",
                function=partial(self.generate_documentation, doc_type=doc_type)
            )
            for doc_type in DocumentationType
        ]

    async def generate_documentation(
        self,
        state: WorkflowState,
        doc_type: DocumentationType
    ) -> str:
        """Generate documentation for a specific type"""
        prompt = self.prompts[doc_type]
        
        # Prepare context
        context = {
            "code": state.code,
            "analysis_level": state.analysis_level.value
        }
        
        # Add analysis result if needed
        if doc_type != DocumentationType.ANALYSIS:
            context["analysis"] = state.results.get(DocumentationType.ANALYSIS, "")
            
        # Generate documentation
        messages = prompt.format_messages(**context)
        response = self.llm.invoke(messages)
        
        return response.content

class DocumentationGraph:
    """Constructs and manages the documentation workflow graph"""
    
    def __init__(
        self,
        tools: DocumentationTools,
        doc_types: Set[DocumentationType],
        analysis_level: AnalysisLevel = AnalysisLevel.DETAILED
    ):
        self.tools = tools
        self.doc_types = doc_types
        self.analysis_level = analysis_level
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the workflow graph"""
        # Create the graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each documentation type
        tool_executor = ToolExecutor(self.tools.create_tools())
        
        for doc_type in self.doc_types:
            workflow.add_node(doc_type.value, tool_executor)
            
        # Add conditional edges
        workflow.add_edge("start", DocumentationType.ANALYSIS.value)
        
        # Add conditional transitions
        for doc_type in self.doc_types:
            if doc_type != DocumentationType.ANALYSIS:
                workflow.add_conditional_edges(
                    doc_type.value,
                    self._get_next_tasks,
                    {
                        next_type.value: next_type.value 
                        for next_type in self.doc_types
                    }
                )
        
        # Set entry and exit
        workflow.set_entry_point("start")
        workflow.set_finish_point("end")
        
        return workflow.compile()
    
    def _get_next_tasks(self, state: WorkflowState) -> List[str]:
        """Determine next tasks based on dependencies"""
        next_tasks = []
        
        for doc_type in self.doc_types:
            task = state.tasks[doc_type]
            if (
                task.status == "pending" 
                and task.dependencies.issubset(state.completed_tasks)
            ):
                next_tasks.append(doc_type.value)
                
        return next_tasks or ["end"]
    
    async def generate(self, code: str) -> WorkflowState:
        """Generate complete documentation"""
        # Initialize state
        initial_state = WorkflowState(
            code=code,
            analysis_level=self.analysis_level,
            tasks={
                doc_type: DocumentationTask(
                    doc_type=doc_type,
                    dependencies=self._get_dependencies(doc_type)
                )
                for doc_type in self.doc_types
            }
        )
        
        # Execute the graph
        final_state = await self.graph.arun(initial_state)
        return final_state
    
    def _get_dependencies(self, doc_type: DocumentationType) -> Set[DocumentationType]:
        """Get dependencies for a documentation type"""
        dependencies = {
            DocumentationType.API_SPEC: {DocumentationType.ANALYSIS},
            DocumentationType.ARCHITECTURE: {DocumentationType.ANALYSIS},
            DocumentationType.SECURITY: {DocumentationType.ANALYSIS},
            DocumentationType.PERFORMANCE: {DocumentationType.ANALYSIS},
            DocumentationType.DOCSTRINGS: {DocumentationType.ANALYSIS},
            DocumentationType.EXAMPLES: {DocumentationType.DOCSTRINGS},
            DocumentationType.TESTS: {DocumentationType.EXAMPLES},
        }
        return dependencies.get(doc_type, set())

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize tools with LLM
        llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        tools = DocumentationTools(llm)
        
        # Configure documentation types
        doc_types = {
            DocumentationType.ANALYSIS,
            DocumentationType.API_SPEC,
            DocumentationType.ARCHITECTURE,
            DocumentationType.SECURITY,
            DocumentationType.DOCSTRINGS,
            DocumentationType.EXAMPLES
        }
        
        # Create documentation graph
        doc_graph = DocumentationGraph(
            tools=tools,
            doc_types=doc_types,
            analysis_level=AnalysisLevel.DETAILED
        )
        
        # Sample code
        sample_code = """
        @app.route('/api/users', methods=['GET'])
        def get_users():
            '''Retrieve list of users with pagination'''
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 10, type=int)
            users = User.query.paginate(page=page, per_page=per_page)
            return jsonify([user.to_dict() for user in users.items])
        """
        
        try:
            # Generate documentation
            final_state = await doc_graph.generate(sample_code)
            
            # Print results
            print("Generated Documentation:")
            for doc_type in doc_types:
                task = final_state.tasks[doc_type]
                print(f"\n{doc_type.value.upper()}:")
                print(f"Status: {task.status}")
                print(f"Content:\n{task.content}")
                if task.metadata:
                    print(f"Metadata: {task.metadata}")
                    
        except Exception as e:
            print(f"Error generating documentation: {e}")

    asyncio.run(main())