from langchain_community.tools import tool

@tool
def multiply(a:float, b:float) -> float:
    """A method that is used to multiply two numbers"""
    return a*b
@tool 
def add(a: float, b:float) -> float:
    """A method that is used to add two numbers"""
    return a*b

class CalculatorToolKit:
    def get_tools():
        return [multiply, add]

for t in CalculatorToolKit.get_tools():
    print(t.name, " ===> ", t.description)