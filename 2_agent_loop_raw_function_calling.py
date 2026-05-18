from dotenv import load_dotenv

load_dotenv()

# from langchain.chat_models import init_chat_model
# from langchain.tools import tool
# from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage

from langsmith import traceable

import ollama

MAX_AGENT_ITERATIONS =10
MODEL ="qwen3:14b"


#-----Tools (custom tool )-----

@traceable(name="Langchain agent loop")
def get_product_price(product:str)->float:
    """Look for proce of the product in catalog"""
    print(f"Checking price of the product {product}")
    price= {"laptop":1299.99,"headphone":149.95,"keyboard":89.50}
    return price.get(product,0)

@traceable(name="Langchain agent loop")
def apply_discount(price:float,discount_tier:str)->float:
    """Apply discount tier to a price and return the final price.
    Available tiers: bronze, silve, gold."""
    print(f"Applying discount tier on the product, price is {price} and discount tier is {discount_tier}")
    discount_percentage={"bronze":5,"silver":12,"gold":23}
    discount= discount_percentage.get(discount_tier,0)
    return round((price*(1-(discount/100))),2)

# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
tools_for_llm =[
    {
        "type":"function",
        "function":{
            "name":"get_product_price",
            "description":"Look for proce of the product in catalog",
            "parameters":{
                "type":"object",
                "properties":{
                    "product":{
                        "type":"string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required":["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)



#--------- Agent Loop-------
@traceable(name="Ollama agent loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    # On device Ollam model
    # llm=init_chat_model(f"ollama:{MODEL}",temperature=0)
    # llm= init_chat_model(f"openai:gpt-5",temperature=0)
    # llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" *60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1,MAX_AGENT_ITERATIONS+1):
        print(f"\n --- Iteration {iteration}---")

        response= ollama_chat_traced(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Model output is {ai_message.content}")
            return ai_message.content
        
        #LLM can return multiplle tools for calling, so I will apply simple logic to pick the first tool call
        # in case it gives multiple tools
        tool_call=tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"Tool selected {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")
        
        observation = tool_to_use(**tool_args)

        print(f"Tool result  {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    
    print(f"We maxed out on iterations without final answer.")
    return None





if __name__ == "__main__":
    print("Hello langchain bind tool agents")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")