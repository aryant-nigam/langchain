from langchain_community.tools import tool

@tool
def multiply(a:float, b:float) -> float:
    """A method that is used to multipy two numbers"""
    return a*b

# res = multiply.invoke({"a":3, "b":5})
# print(res)
# print(multiply.name, multiply.description,multiply.args, multiply.args_schema.model_json_schema(), sep="\n\n")

#=================================================================================

from pydantic import BaseModel, Field
from langchain_community.tools import StructuredTool

class MultiplyInput(BaseModel):
    a: float =  Field(required=True, description="First number to multiply")
    b: float =  Field(required=True, description="Second number to multiply")
    
def multiply(a: float, b: float) -> float:
    return a*b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="multiply",
    description="A tool that is used to multiply two numbers",
    args_schema=MultiplyInput
)

# res = multiply_tool.invoke({"a":3, "b":5})
# print(res)


#=================================================================================

from langchain_community.tools import BaseTool
from typing import Type

class MultiplicationTool(BaseTool):
    name: str = "multiply"
    description: str = "A tool that is use dto multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput
    
    def _run(self, a:float, b: float) -> float:
        return a*b

t =  MultiplicationTool()
# res = t.invoke({"a":3, "b":5})
# print(res)
# print(t.model_json_schema())

