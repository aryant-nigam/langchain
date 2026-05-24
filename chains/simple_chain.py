from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv      
load_dotenv()

prompt = PromptTemplate(
        template = "Generate five interesting facts about the {topic}", 
        input_variables=["topic"]
    )

model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

parser = StrOutputParser()

chain = prompt | model | parser

chain.get_graph().print_ascii()
# chain_response = chain.invoke({"topic": "moon"})

# print(chain_response)