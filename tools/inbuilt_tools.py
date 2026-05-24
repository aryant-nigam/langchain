from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
result= search_tool.invoke("What is one trending news in India right now")
print(result)