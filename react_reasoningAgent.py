from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

# Importing react file
from nodes_tools import llm, tools

load_dotenv()

SYSYEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """

    print(f"run_agent_reasoning")
    response = llm.invoke([{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}

tool_node = ToolNode(tools)