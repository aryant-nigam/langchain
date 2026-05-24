from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv      
load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
embedding = embedding_model.embed_query("Delhi is the capital of India")
print(embedding)

docs_emb = embedding_model.embed_documents(["Delhi is the capital of India", "Mumbai is the financial capital of India"])
print(docs_emb)