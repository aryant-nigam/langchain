from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

docs = [
    Document(
        page_content="Mumbai Indians is one of the most successful teams in the IPL. The team is based in Mumbai and plays its home matches at Wankhede Stadium. It has won multiple IPL titles under captain Rohit Sharma.",
        metadata={"team": "Mumbai Indians", "city": "Mumbai"}
    ),

    Document(
        page_content="Chennai Super Kings is a highly popular IPL franchise based in Chennai. Led for many years by MS Dhoni, the team plays its home games at MA Chidambaram Stadium and has won several IPL championships.",
        metadata={"team": "Chennai Super Kings", "city": "Chennai"}
    ),

    Document(
        page_content="Royal Challengers Bangalore is an IPL team based in Bengaluru. Known for star players like Virat Kohli, the team plays its home matches at M. Chinnaswamy Stadium.",
        metadata={"team": "Royal Challengers Bangalore", "city": "Bengaluru"}
    ),

    Document(
        page_content="Kolkata Knight Riders is an IPL franchise from Kolkata owned by Bollywood actor Shah Rukh Khan. The team plays at Eden Gardens and has won the IPL title multiple times.",
        metadata={"team": "Kolkata Knight Riders", "city": "Kolkata"}
    ),

    Document(
        page_content="Delhi Capitals represents Delhi in the IPL. The team plays its home matches at Arun Jaitley Stadium and has featured players like Rishabh Pant and David Warner.",
        metadata={"team": "Delhi Capitals", "city": "Delhi"}
    )
]

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=64)
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_db",
    collection_name="ipl_teams",
)

# res = vector_store.add_documents(docs)
# print(res)

# docs = vector_store.get(include=["metadatas", "documents", "embeddings"])
# print(docs)

res = vector_store.similarity_search("Which team is based in Mumbai?", k=1)
print(res)