from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  

load_dotenv()

country = input("Enter the country name: ")
template = PromptTemplate(
    template="What is capital of {country}? Also tell the basic information about it in 50 words. Keep a tone like that of a city guide", 
    input_variables=["country"],
    validate_template=True
)

prompt = template.invoke({"country": country})
chat_model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

response = chat_model.invoke(prompt)
print(response.content)