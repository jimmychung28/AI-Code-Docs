from typing import Dict, TypedDict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import re
from langgraph.graph import Graph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from datetime import datetime

class CodeType(Enum):
    API = "api"
    LIBRARY = "library"
    SCRIPT = "script"

class DocSection(Enum):
    EMPTY = ""
    ANALYSIS = "analysis"
    API_REFERENCE = "api_reference"
    AUTHENTICATION = "authentication"
    EXAMPLES = "examples"
    ERRORS = "errors"
    TESTING = "testing"
    COMPLETED = "completed"

@dataclass
class APIEndpoint:
    """Structure for API endpoint documentation"""
    path: str
    method: str
    description: str
    parameters: Dict[str, Dict[str, str]]
    request_body: Optional[Dict[str, any]]
    response: Dict[str, any]
    example_request: str
    example_response: str

@dataclass
class APIDocumentation:
    """Structure for complete API documentation"""
    title: str
    base_url: str
    version: str
    description: str
    authentication: Dict[str, str]
    endpoints: List[APIEndpoint]
    error_codes: Dict[str, str]
    rate_limits: str

@dataclass
class DocumentationOutput:
    """Structure for organized documentation output"""
    api_docs: Optional[APIDocumentation]
    analysis: str
    examples: str
    test_cases: str
    timestamp: str
    is_api: bool
    
    def to_markdown(self) -> str:
        """Convert documentation to formatted markdown with Stripe-like styling"""
        print("is_api",flush=True)
        print(self.is_api,flush=True)
        if not self.is_api:
            return self._generate_library_markdown()
        print(self.api_docs,flush=True)
        return f"""# {self.api_docs.title} API Reference

{self.api_docs.description}

## Base URL
`{self.api_docs.base_url}`

## Authentication
{self._format_authentication()}

## API Endpoints

{self._format_endpoints()}

## Error Codes

{self._format_errors()}

## Rate Limits
{self._format_rate_limits()}

---
Generated on: {self.timestamp}
"""

    def _format_authentication(self) -> str:
        auth = self.api_docs.authentication
        return f"""```bash
# Authentication using API key
curl -X GET "{self.api_docs.base_url}/endpoint" \\
  -H "Authorization: Bearer YOUR_API_KEY"
```

{auth.get('description', '')}
"""

    def _format_endpoints(self) -> str:
        formatted = []
        for endpoint in self.api_docs.endpoints:
            formatted.append(f"""### {endpoint.method.upper()} {endpoint.path}

{endpoint.description}

**Parameters**
{self._format_parameters(endpoint.parameters)}

**Example Request**
```bash
{endpoint.example_request}
```

**Example Response**
```json
{endpoint.example_response}
```
""")
        return "\n".join(formatted)

    def _format_parameters(self, params: Dict[str, Dict[str, str]]) -> str:
        if not params:
            return "No parameters required."
            
        rows = ["| Parameter | Type | Required | Description |",
                "|-----------|------|----------|-------------|"]
        
        for name, details in params.items():
            rows.append(
                f"| `{name}` | {details.get('type', '')} | "
                f"{details.get('required', 'False')} | {details.get('description', '')} |"
            )
        return "\n".join(rows)

    def _format_errors(self) -> str:
        rows = ["| Code | Description |",
                "|------|-------------|"]
        
        for code, desc in self.api_docs.error_codes.items():
            rows.append(f"| `{code}` | {desc} |")
        return "\n".join(rows)

    def _format_rate_limits(self) -> str:
        if not self.api_docs.rate_limits:
            return "No rate limits specified."
            
        # return "\n".join([f"- {k}: {v}" for k, v in self.api_docs.rate_limits])
        return self.api_docs.rate_limits

    def _generate_library_markdown(self) -> str:
        """Generate markdown for non-API code"""
        return f"""# Code Documentation

## Code Analysis
{self.analysis}

## Usage Examples
{self.examples}

## Test Cases
{self.test_cases}

---
Generated on: {self.timestamp}
"""

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    code: str
    documentation: DocumentationOutput
    current_section: DocSection
    error: Optional[str]


class DocumentationGenerator:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
        self.workflow = self._create_workflow()

    def _detect_api_patterns(self, code: str) -> bool:
        """Use regex patterns to detect common API patterns in code"""
        api_patterns = [
            r'@(app|router)\.(get|post|put|delete|patch)',  # Flask/FastAPI patterns
            r'app.use\([\'"]/',  # Express.js patterns
            r'@RequestMapping',  # Spring patterns
            r'class.*Controller',  # Common controller patterns
            r'(res|response)\.(json|send)',  # Response patterns
            r'router\.[A-Za-z]+\([\'"]/',  # Router patterns
        ]
        
        return any(re.search(pattern, code, re.IGNORECASE) for pattern in api_patterns)
    def _analyze_code(self, state: AgentState) -> AgentState:
        """Analyzes the code and determines if it's an API, identifying key components."""
        try:
            messages = [
                SystemMessage(content="""You are a senior API documentation expert. Analyze the given code to:
1. Determine if it's an API implementation (look for route handlers, HTTP methods, request/response handling)
2. Identify API endpoints, methods, and structures
3. Evaluate authentication mechanisms
4. Identify error handling patterns
5. Note any rate limiting or security measures
6. Identify request/response formats

Return your analysis in a structured format starting with "IS_API: true" or "IS_API: false"."""),
                HumanMessage(content=f"Analyze this code:\n{state['code']}")
            ]
            
            response = self.llm.invoke(messages)
            analysis = response.content
            
            # Parse API detection from analysis
            is_api = "is_api: true" in analysis.lower()
            print(analysis.lower())
            print("is_api2",flush=True)
            print(is_api,flush=True)
            if is_api != state["documentation"].is_api:
                # Update documentation structure if API detection differs
                print("is_api3",flush=True)
                print(is_api,flush=True)
                state["documentation"].is_api = is_api
                if is_api:
                    state["documentation"].api_docs = APIDocumentation(
                        title="",
                        base_url="",
                        version="",
                        description="",
                        authentication={},
                        endpoints=[],
                        error_codes={},
                        rate_limits={}
                    )
            
            state["messages"].extend([messages[-1], response])
            state["documentation"].analysis = analysis
            state["current_section"] = DocSection.ANALYSIS.value
        except Exception as e:
            state["error"] = f"Error in code analysis: {str(e)}"
        return state

    def _generate_api_reference(self, state: AgentState) -> AgentState:
        """Generates comprehensive API reference documentation."""
        if not state["documentation"].is_api:
            state["current_section"] = DocSection.EXAMPLES.value
            return state
            
        try:
            messages = [
                SystemMessage(content="""Generate comprehensive API documentation including:
1. Each endpoint's path, method, and purpose
2. Request parameters and body schemas
3. Response formats and status codes
4. Authentication requirements
5. Example requests and responses in curl format
6. Error scenarios and handling
7. Rate limits if applicable"""),
                HumanMessage(content=f"Generate API reference for this code based on the analysis:\n{state['code']}\n\nAnalysis:\n{state['documentation'].analysis}")
            ]
            
            response = self.llm.invoke(messages)
            
            # Replace the ChatPromptTemplate with direct message creation
            messages = [
                SystemMessage(content="""Extract API title, base URL, version, and description from the documentation.
                Return the information in the following Python dictionary format EXACTLY:
                {
                    "title": "API Title",
                    "base_url": "https://api.example.com",
                    "version": "v1",
                    "description": "API description",
                    "authentication": "authentication description",
                    "error_codes": "error codes description",
                    "rate_limits": "rate limits description"
                }"""),
                HumanMessage(content=response.content)
            ]
            
            api_info = self.llm.invoke(messages)

            print("api_info",flush=True)
            print(api_info.content,flush=True)
            # Update the API documentation with basic info
            api_info_dict = eval(api_info.content)
            state["documentation"].api_docs.title = api_info_dict.get("title", "")
            state["documentation"].api_docs.base_url = api_info_dict.get("base_url", "")
            state["documentation"].api_docs.version = api_info_dict.get("version", "")
            state["documentation"].api_docs.description = api_info_dict.get("description", "")
            state["documentation"].api_docs.authentication = api_info_dict.get("authentication", "")
            state["documentation"].api_docs.error_codes = api_info_dict.get("error_codes", "")
            state["documentation"].api_docs.rate_limits = api_info_dict.get("rate_limits", "")
            
            # Parse and update endpoints
            endpoints_messages = [
                SystemMessage(content="""Extract API endpoints information in the following format for each endpoint:
                {
                    "path": "/endpoint",
                    "method": "GET/POST/etc",
                    "description": "description",
                    "parameters": {"param": {"type": "string", "required": true, "description": "desc"}},
                    "request_body": {"type": "object", "properties": {}},
                    "response": {"type": "object", "properties": {}},
                    "example_request": "curl example",
                    "example_response": "json response"
                }"""),
                HumanMessage(content=response.content)
            ]
            try:
                endpoints_response = self.llm.invoke(endpoints_messages)
            except Exception as e:
                print("endpoints_response_2",flush=True)
                print(e,flush=True)
            print("endpoints_response",flush=True)
            print(endpoints_response.content,flush=True)
            # Update endpoints in the documentation
            try:
                # Replace JavaScript boolean values with Python ones
                formatted_content = endpoints_response.content.replace('true', 'True').replace('false', 'False')
                endpoints_list = eval(formatted_content)
            except Exception as e:
                print("endpoints_list_2", flush=True)
                print(e)
            print("endpoints_list",flush=True)
            print(endpoints_list,flush=True)
            print("HII")
            print(type(endpoints_list),flush=True)
            print("WOIJODIWO")
            print([
                APIEndpoint(**endpoint) for endpoint in endpoints_list
            ],flush=True)
            state["documentation"].api_docs.endpoints = [
                APIEndpoint(**endpoint) for endpoint in endpoints_list
            ]

            print("endpoints_list_3",flush=True)
            print(state["documentation"].api_docs.endpoints,flush=True)
            
            # Update the messages history
            state["messages"].extend([messages[-1], response])
            state["current_section"] = DocSection.API_REFERENCE.value
        except Exception as e:
            state["error"] = f"Error in API reference generation: {str(e)}"
        return state

    def _document_authentication(self, state: AgentState) -> AgentState:
        """Documents authentication methods and requirements."""
        if not state["documentation"].is_api:
            state["current_section"] = DocSection.EXAMPLES.value
            return state
            
        try:
            messages = [
                SystemMessage(content="""Generate authentication documentation including:
                1. Authentication methods supported
                2. How to obtain API keys/tokens
                3. How to include authentication in requests
                4. Security best practices
                5. Example requests with authentication"""),
                HumanMessage(content=f"Document authentication for this API:\n{state['code']}\n\nAnalysis:\n{state['documentation'].analysis}")
            ]
            
            response = self.llm.invoke(messages)
            
            # Update authentication documentation
            state["documentation"].api_docs.authentication = {
                "description": response.content,
                "examples": self._extract_auth_examples(response.content)
            }
            
            state["messages"].extend([response])
            state["current_section"] = DocSection.AUTHENTICATION.value
        except Exception as e:
            state["error"] = f"Error in authentication documentation: {str(e)}"
        return state

    def _create_examples(self, state: AgentState) -> AgentState:
        """Creates comprehensive usage examples."""
        try:
            if state["documentation"].is_api:
                example_messages = [
                    SystemMessage(content="""Create API usage examples that:
1. Show complete request/response cycles
2. Include authentication
3. Handle errors and edge cases
4. Use realistic data
5. Cover all main endpoints
6. Include different programming languages"""),
                    HumanMessage(content=f"Create examples for this API:\n{state['code']}")
                ]
            else:
                example_messages = [
                    SystemMessage(content="""Create code examples that:
1. Start with basic usage
2. Progress to advanced scenarios
3. Include error handling
4. Show best practices
5. Use realistic scenarios"""),
                    HumanMessage(content=f"Create examples for this code:\n{state['code']}")
                ]
            response = self.llm.invoke(example_messages)
            
            state["messages"].extend([response])
            state["documentation"].examples = response.content
            state["current_section"] = DocSection.EXAMPLES.value
        except Exception as e:
            state["error"] = f"Error in example generation: {str(e)}"
        return state

    def _document_errors(self, state: AgentState) -> AgentState:
        """Documents error codes, messages, and handling."""
        if not state["documentation"].is_api:
            state["current_section"] = DocSection.TESTING.value
            return state
            
        try:
            # Replace ChatPromptTemplate with direct message creation
            error_messages = [
                SystemMessage(content="""Document API errors including:
                1. All possible error codes
                2. Error messages and meanings
                3. How to handle each error
                4. Example error responses
                5. Best practices for error handling"""),
                HumanMessage(content=f"Document error handling for this API:\n{state['code']}")
            ]
            response = self.llm.invoke(error_messages)
            
            # Replace ChatPromptTemplate with direct message creation
            error_parse_messages = [
                SystemMessage(content="Extract error codes and their descriptions as a dictionary"),
                HumanMessage(content=response.content)
            ]
            error_codes_response = self.llm.invoke(error_parse_messages)
            
            # Update error documentation
            state["documentation"].api_docs.error_codes = eval(error_codes_response.content)
            
            state["messages"].extend([response])
            state["current_section"] = DocSection.ERRORS.value
        except Exception as e:
            state["error"] = f"Error in error documentation: {str(e)}"
        return state

    def _generate_test_cases(self, state: AgentState) -> AgentState:
        """Generates comprehensive test cases."""
        try:
            if state["documentation"].is_api:
                test_messages = [
                    SystemMessage(content="""Generate API test cases that:
1. Test all endpoints
2. Include authentication tests
3. Cover error scenarios
4. Test rate limiting
5. Include integration tests
6. Use pytest fixtures and markers"""),
                    HumanMessage(content=f"Generate test cases for this API:\n{state['code']}")
                ]
            else:
                test_messages = [
                    SystemMessage(content="""Generate test cases that:
1. Include unit tests
2. Cover edge cases
3. Test error handling
4. Use appropriate test fixtures
5. Include integration tests if applicable"""),
                    HumanMessage(content=f"Generate test cases for this code:\n{state['code']}")
                ]
            response = self.llm.invoke(test_messages)
            
            state["messages"].extend([response])
            state["documentation"].test_cases = response.content
            state["current_section"] = DocSection.TESTING.value
        except Exception as e:
            state["error"] = f"Error in test case generation: {str(e)}"
        return state

    def _end_documentation(self, state: AgentState) -> AgentState:
        """Finalizes the documentation and performs any cleanup."""
        try:
            # Add timestamp
            state["documentation"].timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Validate documentation completeness
            if state["documentation"].is_api:
                self._validate_api_documentation(state["documentation"])
            
            state["current_section"] = DocSection.COMPLETED.value
        except Exception as e:
            state["error"] = f"Error in documentation finalization: {str(e)}"
        return state

    def _extract_auth_examples(self, content: str) -> Dict[str, str]:
        """Helper method to extract authentication examples from content."""
        # Implementation to parse authentication examples
        return {}

    def _validate_api_documentation(self, documentation: DocumentationOutput) -> None:
        """Helper method to validate API documentation completeness."""
        if not documentation.is_api:
            return
        print("validate",flush=True)
        print(documentation.api_docs,flush=True)
        required_fields = [
            documentation.api_docs.title,
            documentation.api_docs.base_url,
            documentation.api_docs.version,
            documentation.api_docs.description,
            documentation.api_docs.authentication,
            documentation.api_docs.endpoints,
            documentation.api_docs.error_codes
        ]
        
        if any(not field for field in required_fields):
            
            raise ValueError("API documentation is incomplete. Missing required fields.")
    def _should_continue(self, state: AgentState) -> str:
        """
        Determines the next section in the documentation workflow based on current state.
        
        This method implements the workflow logic for both API and non-API documentation,
        ensuring all necessary sections are completed in the correct order.
        
        Args:
            state (AgentState): Current state of the documentation process
            
        Returns:
            str: The next section value from DocSection enum
            
        Logic:
        1. If there's an error, move to completion
        2. For APIs: analysis -> api_reference -> authentication -> examples -> errors -> testing -> completed
        3. For non-APIs: analysis -> examples -> testing -> completed
        4. Skip API-specific sections for non-API code
        """
        # Handle errors by moving to completion
        if state.get("error"):
            return DocSection.COMPLETED.value
        
        # Get current section
        current = DocSection(state["current_section"])
        
        # Define workflow paths
        api_workflow = {
            DocSection.EMPTY: DocSection.ANALYSIS,
            DocSection.ANALYSIS: DocSection.API_REFERENCE,
            DocSection.API_REFERENCE: DocSection.AUTHENTICATION,
            DocSection.AUTHENTICATION: DocSection.EXAMPLES,
            DocSection.EXAMPLES: DocSection.ERRORS,
            DocSection.ERRORS: DocSection.TESTING,
            DocSection.TESTING: DocSection.COMPLETED
        }
        
        non_api_workflow = {
            DocSection.EMPTY: DocSection.ANALYSIS,
            DocSection.ANALYSIS: DocSection.EXAMPLES,
            DocSection.EXAMPLES: DocSection.TESTING,
            DocSection.TESTING: DocSection.COMPLETED
        }
        
        # Check for required sections completion
        def _is_section_complete(section: DocSection) -> bool:
            """Helper function to check if a section is complete"""
            if section == DocSection.API_REFERENCE:
                return bool(state["documentation"].api_docs.endpoints)
            elif section == DocSection.AUTHENTICATION:
                return bool(state["documentation"].api_docs.authentication)
            elif section == DocSection.EXAMPLES:
                return bool(state["documentation"].examples)
            elif section == DocSection.ERRORS:
                return bool(state["documentation"].api_docs.error_codes)
            elif section == DocSection.TESTING:
                return bool(state["documentation"].test_cases)
            return True

        # Select appropriate workflow
        workflow = api_workflow if state["documentation"].is_api else non_api_workflow
        
        # Handle section transitions
        next_section = workflow.get(current, DocSection.COMPLETED)
        
        # If current section is not complete, stay on it
        if not _is_section_complete(current):
            return current.value
        
        # Special handling for API-specific sections in non-API code
        if not state["documentation"].is_api:
            if next_section in [DocSection.API_REFERENCE, DocSection.AUTHENTICATION, DocSection.ERRORS]:
                return DocSection.EXAMPLES.value
        
        # Validation before completion
        if next_section == DocSection.COMPLETED:
            # For APIs, ensure all required sections are complete
            if state["documentation"].is_api:
                required_sections = [
                    DocSection.API_REFERENCE,
                    DocSection.AUTHENTICATION,
                    DocSection.EXAMPLES,
                    DocSection.ERRORS,
                    DocSection.TESTING
                ]
                
                # Find first incomplete required section
                for section in required_sections:
                    if not _is_section_complete(section):
                        return section.value
                        
            # For non-APIs, ensure examples and testing are complete
            else:
                required_sections = [DocSection.EXAMPLES, DocSection.TESTING]
                for section in required_sections:
                    if not _is_section_complete(section):
                        return section.value
        
        return next_section.value

    def _create_workflow(self) -> Graph:
            """Creates the workflow graph for documentation generation"""
            workflow = Graph()
            
            # Add nodes for API documentation
            workflow.add_node("analyze", self._analyze_code)
            workflow.add_node("api_reference", self._generate_api_reference)
            workflow.add_node("authentication", self._document_authentication)
            workflow.add_node("examples", self._create_examples)
            workflow.add_node("errors", self._document_errors)
            workflow.add_node("testing", self._generate_test_cases)
            workflow.add_node("end", self._end_documentation)
            
            # Set entry point
            workflow.set_entry_point("analyze")
            
            # Add edges with conditional routing based on documentation section
            workflow.add_conditional_edges(
                "analyze",
                self._should_continue,
                {
                    DocSection.API_REFERENCE.value: "api_reference",
                    DocSection.AUTHENTICATION.value: "authentication",
                    DocSection.EXAMPLES.value: "examples",
                    DocSection.ERRORS.value: "errors",
                    DocSection.TESTING.value: "testing",
                    DocSection.COMPLETED.value: "end"
                }
            )
            
            # Add edges for remaining nodes
            for node in ["api_reference", "authentication", "examples", "errors", "testing"]:
                workflow.add_conditional_edges(
                    node,
                    self._should_continue,
                    {
                        DocSection.API_REFERENCE.value: "api_reference",
                        DocSection.AUTHENTICATION.value: "authentication",
                        DocSection.EXAMPLES.value: "examples",
                        DocSection.ERRORS.value: "errors",
                        DocSection.TESTING.value: "testing",
                        DocSection.COMPLETED.value: "end"
                    }
                )
            
            workflow.set_finish_point("end")
            return workflow.compile()

    def generate(self, code: str) -> DocumentationOutput:
        """
        Generates comprehensive documentation for the given code.
        
        Args:
            code (str): The source code to document
            
        Returns:
            DocumentationOutput: Generated documentation including API reference,
                               examples, and test cases if it's an API, or regular
                               documentation if it's a library
        """
        is_api = self._detect_api_patterns(code)
        
        initial_state = {
            "messages": [],
            "code": code,
            "documentation": DocumentationOutput(
                api_docs=APIDocumentation(
                    title="",
                    base_url="",
                    version="",
                    description="",
                    authentication={},
                    endpoints=[],
                    error_codes={},
                    rate_limits={}
                ) if is_api else None,
                analysis="",
                examples="",
                test_cases="",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                is_api=is_api
            ),
            "current_section": DocSection.EMPTY.value,
            "error": None
        }
        
        final_state = self.workflow.invoke(initial_state)
        
        if final_state.get("error"):
            raise RuntimeError(f"Documentation generation failed: {final_state['error']}")
            
        return final_state["documentation"]

# Example usage
if __name__ == "__main__":
    sample_api_code = """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    app = FastAPI()
    
    class Item(BaseModel):
        name: str
        price: float
    
    @app.post("/items/")
    async def create_item(item: Item):
        return {"id": 1, **item.dict()}
    
    @app.get("/items/{item_id}")
    async def get_item(item_id: int):
        if item_id < 0:
            raise HTTPException(status_code=400, detail="Invalid ID")
        return {"id": item_id, "name": "Sample", "price": 99.9}
    """
    
    generator = DocumentationGenerator()
    documentation = generator.generate(sample_api_code)
    print(documentation.to_markdown())