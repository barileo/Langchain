import asyncio

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage


load_dotenv()

async def main():
    print(f"running...")
    client= MultiServerMCPClient(
        {
            "math":{
                "command":"python",
                "args":["/Users/barinderpaulsingh/Projects/UdemyLangChainAgenticAI/langchain-course/servers/math_server.py"],
                "transport":"stdio",
            },
            "weather":{
                "transport":"streamable_http",
                "url":"http://127.0.0.1:8000/mcp"
            }
        }
    )
    tools= await client.get_tools()
    llm= ChatOpenAI()
    agent=create_agent(model=llm,tools=tools)

    math_response = await agent.ainvoke({
        # "messages": [HumanMessage(content="what is (4+5)*3?")]
        "messages": [HumanMessage(content="weather in Seattle")]
    })

    print(math_response["messages"][-1].content)



if __name__=="__main__":
    asyncio.run(main())

