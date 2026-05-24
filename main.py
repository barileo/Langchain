import os
from dotenv import load_dotenv

load_dotenv()


from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import reflection_chain,generation_chain

class MessagesGraph(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

GENERATE ="generate"
REFLECT= "reflect"

def generation_node(state:MessagesGraph):
    return {"messages":[generation_chain.invoke({"messages":state["messages"]})]}

def reflection_node(state:MessagesGraph):
    res= reflection_chain.invoke({"messages":state["messages"]})
    return {"messages":[HumanMessage(content=res.content)]}

def should_continue(state:MessagesGraph):
    if len(state["messages"])>6:
        return END
    return REFLECT

flow= StateGraph(state_schema=MessagesGraph)

flow.add_node(GENERATE,generation_node)
flow.set_entry_point(GENERATE)
flow.add_node(REFLECT,reflection_node)
flow.add_edge(REFLECT,GENERATE)
flow.add_conditional_edges(GENERATE,should_continue,path_map={END:END,REFLECT:REFLECT})

graph = flow.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()






if __name__ == "__main__":
    print("Hello from langchain-course!")
    print(os.getenv("LANGSMITH_API_KEY"))
    inputs={
        "messages":[
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    finaleresult=graph.invoke(input=inputs)
    print(f"Final Result: {finaleresult["messages"][-1].content}")
