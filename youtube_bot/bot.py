from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatMessagePromptTemplate, AIMessagePromptTemplate, PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

class YoutubeBot:
    def __init__(self):
        self.vector_store = None
        
    def fetch_video_transcript(self, video_id: str):
        try:
            if not video_id:
                raise Exception("video_id can't be empty")

            ytt = YouTubeTranscriptApi()

            transcript_list = ytt.list(video_id)

            # find the generated Hindi transcript
            transcript = transcript_list.find_generated_transcript(['en'])

            data = transcript.fetch()
            transcript_text = " ".join(chunk.text for chunk in data)

            return transcript_text

        except TranscriptsDisabled:
            print("Transcripts have been disabled for this video")

        except Exception as e:
            print(e)
            print("EXITING: Invalid video id")

    def split_text(self, text:str):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.create_documents([text])
    
    def save_to_vector_store(self, documents):
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = FAISS.from_documents(documents=documents,embedding=embedding_model)
        
if __name__ == "__main__":
    bot = YoutubeBot()
    # FuqNluMTIR8
    transcript = bot.fetch_video_transcript(video_id="Gfr50f6ZBvo")
    docs = bot.split_text(transcript)
    bot.save_to_vector_store(docs)
    retriver = bot.vector_store.as_retriever(search_type="similarity", search_kwargs = {"k":4})
    

    template = PromptTemplate(template="""You are a helpful assistant.
                              Answer only from the provided context.
                              If the context is insufficient just say cant answer.
                              
                              context :{context}
                              query: {query}""", input_variables=["context", "query"])
    res = retriver.invoke("What is deepmind?")
    context = "\n\n".join([doc.page_content for doc in res])
    final_prompt = template.invoke({"context": context, "query": "What is deepmind?"  })
    
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    ans = llm.invoke(final_prompt)
    print(ans)
    
    