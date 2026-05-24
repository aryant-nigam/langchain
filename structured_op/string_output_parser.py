from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

prompt_template_1 = PromptTemplate(template="write a detailed description of the topic in 200-250 words: {topic}", input_variables=["topic"])
prompt_template_2 = PromptTemplate(template="write a summary of the text in 50-60 words: {description}", input_variables=["description"])
model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

def manual_invoke():
    prompt1 = prompt_template_1.invoke("The Eiffel Tower")


    response = model.invoke(prompt1)

    print("detailed Description:\n", response.content)
    
    prompt2 = prompt_template_2.invoke({"description": response.content})
    response_1 = model.invoke(prompt2)

    print("summarised Description:\n", response_1.content)
    
def chained_invokation():
    str_op_parser = StrOutputParser()

    chain = prompt_template_1 | model | str_op_parser | prompt_template_2 | model | str_op_parser
    final_response = chain.invoke({"topic": "The Eiffel Tower"})
    print("Final Summarised Description:\n", final_response)
    

if __name__ == "__main__":
    print("Manual Invokation:")
    manual_invoke()
    # print("\n\nChained Invokation:")
    # chained_invokation()