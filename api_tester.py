import json
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.messages import AIMessage
from typing import Literal
from IPython.display import Image, display
import os
import PIL
import io

memory = MemorySaver()
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
create_api_spec_agent = create_react_agent(
        llm, tools=[], prompt="""Your role is to create an example API spec. The api spec will be used to then create code for the API. 
        The API spec needs to at least have 4 endpoints - 2 GET and 2 POST endpoints. The API spec should be in JSON format. Create an API that someone would potentially use. 
        For example, you should not name anything "test" or "example". Base the API theme on something that you would use in your daily life. The user may or may not pass in a theme for the API.
        """
    )

create_api_code_agent = create_react_agent(llm, tools=[], prompt="""Your role is to write the Python code for an API based on the given API specification.
The API spec will be provided to you in JSON format.
You must generate code using the FastAPI framework. Ensure the code is runnable and correctly implements the endpoints defined in the spec.
RETURN ONLY the complete Python code without any additional explanation. Wrap the code in ```python``` code block.
DO NOT RETURN ANYTHING ELSE. DO NOT RETURN ANY EXPLANATION. DO NOT RETURN THE API SPEC. RETURN ONLY THE API CODE. ensure that the code is properly closed by the ```python``` code block""")

create_api_tests_agent = create_react_agent(llm, tools=[],     prompt="""Your role is to create pytest tests for a FastAPI application.
    The API code will be provided to you. Create comprehensive tests that:
    1. Test each endpoint's success case
    2. Test error cases and edge cases
    3. Use pytest fixtures where appropriate
    4. Include proper assertions
    5. Test both GET and POST endpoints
    6. Use pytest.mark.asyncio for async tests
    7. Include proper imports and setup
    
    The tests should use:
    - pytest
    - FastAPI.testclient
    - proper async/await syntax
    - meaningful test names and descriptions
    
    RETURN ONLY the test code without any additional explanation. Wrap the code in ```python``` code block. ensure that the code is properly closed by the ```python``` code block""")

create_refinement_agent = create_react_agent(
    llm, tools=[], prompt="""Your role is to refine the code. 
    The code should be in python and use the FastAPI framework. 
    The code must be able to run. 
    Identify and fix any issues in the code.
    If the code is already correct, return it as is.
    If the code is testing code, ensure that it is properly formatted and includes all necessary imports.
    If it is testing code, you will be given the API code to reference as well.
    RETURN ONLY the test code without any additional explanation. Wrap the code in ```python``` code block. ensure that the code is properly closed by the ```python``` code block.
    """)

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    
    from_create_code_node: bool  # Track the transition source
def create_code_node(state: State) -> Command[Literal["__end__"]]:
    state["from_create_code_node"] = True
    print("Debug create code - State from_create_code_node", state["from_create_code_node"])  # Add debug print
    try:
        result = create_api_code_agent.invoke(state)
        code_content = result["messages"][-1].content.strip()
        # Save the code to a file
        save_api_code(code_content, "generated_api.py")
    except Exception as e:
        print(f"Error during create_code_node invoke: {e}")
        return Command(
            update={
                "messages": [
                    AIMessage(content=f"Error generating code: {str(e)}", name="create_code_node")
                ]
            },
            goto="__end__",
        )
    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="create_code_node")
            ]
        },
        goto="create_refinement_node",
    )

def create_api_spec_node(state: State) -> Command[Literal["__end__"]]:
    print("Debug create spec - State received.")  # Add debug print
    try:
        result = create_api_spec_agent.invoke(state)
        content = result["messages"][-1].content.strip()
    except Exception as e:
        print(f"Error during create_api_spec_node invoke: {e}")
        return Command(
            update={
                "messages": [
                    AIMessage(content=f"Error generating API spec: {str(e)}", name="create_api_spec_node")
                ]
            },
            goto="__end__",
        )
    return Command(
        update={
            "messages": [
                AIMessage(content=content, name="create_api_spec_node")
            ]
        },
        goto="create_code_node",
    )

def create_api_tests_node(state: State) -> Command[Literal["__end__"]]:
    print("Debug create tests - State received.")  # Add debug print
    
    # Read the generated API code
    try:
        with open("generated_api.py", "r") as f:
            api_code = f.read()
            
        # Update state with API code
        state["messages"].append({
            "role": "user",
            "content": f"Generate pytest tests for this FastAPI code:\n\n{api_code}".strip()
        })
        
        # Generate tests
        try:
            result = create_api_tests_agent.invoke(state)
            test_content = result["messages"][-1].content.strip()
            
            # Save the tests
            save_api_code(test_content, "generated_api_tests.py")
            
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content=f"API tests have been generated and saved to generated_api_tests.py\n\n{test_content}",
                            name="create_api_tests_node"
                        )
                    ]
                },
                goto="__end__",
            )
        except Exception as e:
            print(f"Error during create_api_tests_node invoke: {e}")
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content=f"Error generating tests: {str(e)}",
                            name="create_api_tests_node"
                        )
                    ]
                },
                goto="__end__",
            )
    
    except Exception as e:
        print(f"Error reading generated API code: {e}")
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=f"Error reading generated API code: {str(e)}",
                        name="create_api_tests_node"
                    )
                ]
            },
            goto="__end__",
        )

def create_refinement_node(state: State) -> Command[Literal["__end__"]]:
    print("Debug refinement - State received.")
    try:
        result = create_refinement_agent.invoke(state)
        code_content = result["messages"][-1].content.strip()
        
        if state["from_create_code_node"]:
            save_api_code(code_content, "generated_api.py")
            state["from_create_code_node"] = False
            next_node = "create_api_tests_node"
        else:
            save_api_code(code_content, "generated_api_tests.py")
            next_node = "__end__"
    except Exception as e:
        print(f"Error during create_refinement_node invoke: {e}")
        return Command(
            update={
                "messages": [
                    AIMessage(content=f"Error refining code: {str(e)}", name="create_refinement_node")
                ]
            },
            goto="__end__",
        )

    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content.strip(), name="create_refinement_node")
            ]
        },
        goto=next_node,
    )

def save_api_code(code_content: str, filename: str = "generated_api.py"):
    try:
        # Clean up the code content (remove markdown if present)
        if "```python" in code_content:
            code_content = code_content.split("```python")[1].split("```")[0].strip()
        elif "```" in code_content:
            code_content = code_content.split("```")[1].strip()
            
        # Save to file
        with open(filename, "w") as f:
            f.write(code_content)
        print(f"API code saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving API code: {e}")
        return False

def build_graph():
    graph_builder = StateGraph(State)
    # Add nodes
    graph_builder.add_node("create_api_spec_node", create_api_spec_node)
    graph_builder.add_node("create_code_node", create_code_node)
    graph_builder.add_node("create_refinement_node", create_refinement_node)
    graph_builder.add_node("create_api_tests_node", create_api_tests_node)
    
    # Add edges
    graph_builder.add_edge(START, "create_api_spec_node")
    graph_builder.add_edge("create_api_spec_node", "create_code_node")
    graph_builder.add_edge("create_code_node", "create_refinement_node")
    graph_builder.add_edge("create_refinement_node", "create_api_tests_node")
    graph_builder.add_edge("create_api_tests_node", END)
    graph_builder.add_edge("create_refinement_node", END)
    

    graph = graph_builder.compile(checkpointer=memory)
    return graph

def stream_graph_updates(user_input: str, graph: CompiledStateGraph, config= {"configurable": {"thread_id": "1"}}):
    initial_state = {
    "messages": [
        {
            "role": "user", 
            "content": user_input,
            "type": "human"  # Add type field
        }
    ],
    "from_create_code_node": False  # Initialize key
    }
    try:
        for event in graph.stream(initial_state, config):
            print("Debug - Event:", event)  # Add debug print
            for key, value in event.items():
                print(f"Debug - Processing key: {key}")  # Add debug print
                if "messages" in value and value["messages"]:
                    for msg in value["messages"]:
                        if hasattr(msg, 'content'):
                            print("Assistant:", msg.content)
    except Exception as e:
        print(f"Error during stream: {str(e)}")
        import traceback
        print(traceback.format_exc())  # Add detailed error trace

if __name__ == "__main__":
    # Build the graph
    graph = build_graph()

    # Start the conversation
    print("Welcome to the LangGraph API Tester!")
    print("Type 'quit' to exit.")
    print("Provide an API schema and I will create and run tests for you.")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input, graph)
        except:
            # fallback if input() is not available
            user_input = "What can you help me with?"
            print("User: " + user_input)
            stream_graph_updates(user_input, graph)
            break
