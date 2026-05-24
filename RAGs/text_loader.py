from langchain_community.document_loaders import TextLoader
import os 



loader = TextLoader(f"{os.path.dirname(__file__)}/cricket.txt", encoding="utf-8")

document = loader.load()

print(document)