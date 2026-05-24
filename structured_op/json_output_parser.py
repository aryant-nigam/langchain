from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

parser = JsonOutputParser()
prompt = PromptTemplate(
    template="Give the name,age and city of a fictional charecter {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

def manual_invokation():
    response = model.invoke(prompt.invoke({}))

    response = parser.parse(response.content)

    print(response)

def chained_invokation():
    chain = prompt | model | parser
    final_response = chain.invoke({})
    print(final_response)

if __name__ == "__main__":
    # print("Manual Invokation:")
    # manual_invokation()
    print("\n\nChained Invokation:")
    chained_invokation()
                        
