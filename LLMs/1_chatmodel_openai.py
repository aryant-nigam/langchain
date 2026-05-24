from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.9)

res = model.invoke("What is the current AQI of New Delhi?")
print(res)