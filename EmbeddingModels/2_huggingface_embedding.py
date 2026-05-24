from langchain_huggingface import  HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
load_dotenv()

hfe = HuggingFaceEndpointEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2", task="feature-extraction")

vector = hfe.embed_query("Delhi is the capital of India")
print(vector)