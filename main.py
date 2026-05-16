from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    tTool that searched over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    # return "Seattle weather is sunny"
    return tavily.search(query=query)


llm= ChatOpenAI(model="gpt-5")
tools =[search]
agent=create_agent(model=llm,tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="search top 3 job posting in Seattle for AI engineer. Look on Dice.com, Indeed and LinkedIn")})
    print(result)


if __name__ == "__main__":
    main()
