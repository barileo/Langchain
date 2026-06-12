import asyncio
import os

from dotenv import load_dotenv
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

llm =ChatOpenAI()
studio_server_params = StdioServerParameters(
    command="python",
    args=["/Users/barinderpaulsingh/Projects/UdemyLangChainAgenticAI/langchain-course/servers/math_server.py"],
)

async def main():
    print("Hello from langchain-course!")
    # print(f"{os.getenv("OPENAI_API_KEY")}")
    async with stdio_client(studio_server_params) as (read,write):
        async with ClientSession(read_stream=read,write_stream=write) as session:
            await session.initialize()
            print(f"Session initialized")
            # tools= await session.list_tools()
            tools = await load_mcp_tools(session)
            print(tools)
            agent = create_agent(llm,tools=tools)

            result = await agent.ainvoke({
                "messages": [HumanMessage(content="what is (4+5)*3?")]
            })

            print(result["messages"][-1].content)



if __name__ == "__main__":
    asyncio.run(main())
