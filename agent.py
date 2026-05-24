from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()


chat_model = ChatOpenAI(model="gpt-4o-mini")

agent = create_agent(
    model=chat_model,
    tools=[DuckDuckGoSearchRun()],
    debug=True
)

res = agent.invoke({"messages":[HumanMessage(content="What are the three ways to reach goa from delhi?")]})

for msg in res["messages"]:
    print(msg, end="\n\n ************************")