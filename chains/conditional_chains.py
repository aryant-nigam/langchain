from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableBranch, RunnableLambda

from typing import Literal
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

class FeedbackAnalysis(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback categorized into positive, negative.")
    summary: str = Field(description="A brief summary of the feedback.")

feedback_parser = PydanticOutputParser(pydantic_object=FeedbackAnalysis)

feedback_analysis_prompt = PromptTemplate(
    template="Analyze the following feedback to generate the  brief summary and sentiment of it. Feedback: {feedback}. {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": feedback_parser.get_format_instructions()}
)

positive_feedback_response_prompt = PromptTemplate(
    template="Generate a thanking response to the following feedback: {feedback}",
    input_variables=["feedback"]
)

negative_feedback_response_prompt = PromptTemplate(
    template="Generate an apologetic response to the following feedback: {feedback}. Use sender_name: {sender_name} and brand name: {brand_name} in response email signature.",
    input_variables=["feedback"], 
    partial_variables={"sender_name": "Aryant", "brand_name": "TechSolutions" }
)

feedbacks = [
    """This product exceeded my expectations in both quality and performance. The design is sleek and user-friendly, making it easy to set up and operate right out of the box. The materials feel durable and well-made, which gives confidence in its long-term reliability. Performance-wise, it delivers consistent and efficient results, clearly reflecting thoughtful engineering and attention to detail. The features are practical and genuinely useful rather than unnecessary additions. Overall, it provides excellent value for the price and stands out compared to similar products on the market. I would confidently recommend it to others looking for reliability and quality.""",
    """The product performs its basic functions as expected and meets standard quality levels. The design is simple and functional, though not particularly distinctive. Setup was straightforward, and the instructions were clear enough to follow. While it works adequately for everyday use, it does not offer many standout features that differentiate it from competing products. The materials appear reasonably durable, but long-term reliability remains to be seen. Overall, it provides fair value for its price and fulfills its intended purpose. With some improvements in performance or additional features, it could become more competitive in the market.""",
    """The product did not meet expectations in terms of quality and performance. The design feels somewhat fragile, and the materials do not seem durable enough for long-term use. Setup was more complicated than necessary, with instructions that lacked clarity. During use, the performance was inconsistent, and certain features did not function as smoothly as advertised. Considering the price, the overall value is disappointing compared to similar alternatives available in the market. Improvements in build quality, reliability, and user experience are needed. Without significant refinement, it would be difficult to recommend this product to others seeking dependable performance."""
]

analysis_chain = feedback_analysis_prompt | model | feedback_parser
response_chain = RunnableBranch(
    (lambda analysis: analysis.sentiment == "positive", positive_feedback_response_prompt | model | StrOutputParser()),
    (lambda analysis: analysis.sentiment == "negative", negative_feedback_response_prompt | model | StrOutputParser()),
    RunnableLambda(lambda analysis: "Could not determine sentiment, unable to generate response.")
)

final_chain = analysis_chain | response_chain

final_chain.get_graph().print_ascii()

for i, feedback in enumerate(feedbacks):
    res = final_chain.invoke({"feedback": feedback})
    print(f"Feedback {i}: {feedback[:100]}...")
    print(f"Response: {res}")
    print("-" * 100)