import os
from dotenv import load_dotenv

load_dotenv()


from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph import END,StateGraph
from langgraph.graph.message import add_messages

from chains import reflection_chain,generation_chain

class MessagesGraph(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def main():
    print("Hello from langchain-course!")
    print(os.getenv("LANGSMITH_PROJECT"))


if __name__ == "__main__":
    main()
