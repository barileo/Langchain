from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage
from langsmith import traceable

MAX_AGENT_ITERATIONS =10
MODEL ="qwen3:14b"


#-----Tools (langchain @tool decorator)-----

@tool
def get_product_price(product:str)->float:
    """Look for proce of the product in catalog"""
    print(f"Checking price of the product {product}")
    price= {"laptop":1299.99,"headphone":149.95,"keyboard":89.50}
    return price.get(product,0)

@tool
def get_product_discount(price:float,discount_tier:str)->float:
    """Apply discount tier to a price and return the final price.
    Available tiers: bronze, silve, gold."""
    print(f"Applying discount tier on the product, price is {price} and discount tier is {discount_tier}")
    discount_percentage={"bronze":5,"silver":12,"gold":23}
    discount= discount_percentage.get(discount_tier,0)
    return round((price*(1-(discount/100))),2)


#--------- Agent Loop-------
@traceable(name="Langchain agent loop")
def run_agent(question: str):
    pass


if __name__ == "__main__":
    print("Hello langchain bind tool agents")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")