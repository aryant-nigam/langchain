from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.tools import tool
from langchain_core.messages import HumanMessage


load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

@tool
def multiply(a:float, b:float) -> float:
    """A method that is used to multiply two numbers"""
    return a*b

model_with_tools = model.bind_tools([multiply])

query = HumanMessage("What's 8 multiplied by 2")
messages = [query]

result = model_with_tools.invoke(messages)
messages.append(result)

tool_res = multiply.invoke(result.tool_calls[0])
messages.append(tool_res)

final_op = model_with_tools.invoke(messages)
print(final_op)