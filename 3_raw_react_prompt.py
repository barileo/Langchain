import re
import inspect
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
    price=float(price)
    print(f"Applying discount tier on the product, price is {price} and discount tier is {discount_tier}")
    discount_percentage={"bronze":5,"silver":12,"gold":23}
    discount= discount_percentage.get(discount_tier,0)
    return round((price*(1-(discount/100))),2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

# CHANGE 3: Delete the JSON schemas. Tools now live inside the prompt as plain text.
# We derive descriptions from the functions themselves using inspect.

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable adds *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature= inspect.signature(original_function)
        docstring=inspect.getdoc(tool_function)
        descriptions.append(f"{tool_name}{signature}-{docstring}")
    return "\n".join(descriptions)

tool_description = get_tool_descriptions(tools)
tool_names=", ".join(tools.keys())

react_prompt=f"""
Answer the following questions as best you can. You have access to the following tools:

{tool_description}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


#--------- Agent Loop-------
@traceable(name="Ollama agent loop")
def run_agent(question: str):

    print(f"Question: {question}")
    print("=" *60)
    prompt = react_prompt.format(question=question)
    scratchpad=""


    for iteration in range(1,MAX_AGENT_ITERATIONS+1):
        print(f"\n --- Iteration {iteration}---")
        full_prompt = prompt + scratchpad

        response= ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
            )
        
        output = response.message.content
        print(f"LLM Output:\n{output}")

        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer
        


        # CHANGE 6: Parse tool calls from raw text with regex — fragile if LLM doesn't follow format.
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")


#my Qwen model was gigning me different output therefore I had to change the regular expression 
        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        # raw_args = [x.strip() for x in tool_input_raw.split(",")]
        # args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]
        args = re.findall(r"[\d.]+|[a-zA-Z]+", tool_input_raw)

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))


        print(f"  [Tool Result] {observation}")

        # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"


    print("ERROR: Max iterations reached without a final answer")
    return None





if __name__ == "__main__":
    print("Hello langchain bind tool agents")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")