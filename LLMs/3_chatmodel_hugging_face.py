from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os
load_dotenv()

# using inference API
def user_inderence_api():
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3.2",
        task="text-generation"
    )

    model = ChatHuggingFace(llm=llm)
    result = model.invoke("What is the current AQI of New Delhi?")
    print(result)
    
def user_local_model():
    os.environ["HF_HOME"] = "/Users/aryantnigam/Desktop/2026/langchain_models/huggingface_cache"
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={"temperature": 0.9, "max_new_tokens": 100}
    )
    model = ChatHuggingFace(llm=llm)
    result = model.invoke("What is the capital of Australia?")
    print(result)
    
if __name__ == "__main__":
    # user_inderence_api()
    user_local_model()
    