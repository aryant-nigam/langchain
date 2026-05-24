from typing import TypedDict, Annotated, Optional, Literal

class Review(TypedDict):
    summary : Annotated[str, "A concise summary of the review in one or two sentences."]
    sentiment : Annotated[str, "A sentiment of the review: positive, negative, or neutral."]
    
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

reviews = [
    "I’ve been using this mixer grinder for 6 months now and it works perfectly. The motor is powerful and grinds masalas very smoothly. Even coconut chutney comes out fine without chunks. Totally worth the price.",
    "Motor is strong but the machine shakes a lot while running. Also, the lid doesn’t lock as tightly as I would like. It works, but I was expecting better finishing.",
    "It’s decent but not amazing. Grinding dry spices is great, but when making batter, I need to run it longer than expected. Build quality is okay — jars feel slightly lightweight."
]

chat_model = ChatOpenAI(model="gpt-5-nano-2025-08-07")
# structured_chat_model = chat_model.with_structured_output(Review)
# response = structured_chat_model.invoke(reviews[2])

# print(response)

complex_review = """
I’ve been using the Samsung Galaxy S23 FE for about six weeks now, and overall it’s been a reliable daily driver with a few minor compromises. The Exynos processor inside feels fast for everyday use — apps open quickly, scrolling is smooth, and multitasking between social media, streaming, and light gaming is handled well. However, during longer gaming sessions, the phone does warm up slightly, though not to an alarming level.

The display quality is definitely one of its highlights. The AMOLED screen is bright, vibrant, and sharp, with excellent contrast that makes videos and photos pop. Outdoor visibility is strong, and the 120Hz refresh rate makes everything feel fluid. It’s not quite flagship-tier brightness like the ultra-premium models, but it’s very close.

Battery life has been solid for me — I comfortably get through a full day with moderate use, including browsing, messaging, and some video streaming. Fast charging helps when I need a quick top-up, though it’s not the fastest in its segment.

The camera performance is good overall. Daylight photos come out detailed with natural colors, and the main sensor performs consistently well. Low-light photography is decent but can introduce some noise, and the ultra-wide camera is usable though slightly softer in comparison. The selfie camera does a good job for social media but occasionally smooths skin tones more than necessary.

Build quality feels premium with a sturdy frame, though the glossy back does attract fingerprints. The stereo speakers are clear but could have slightly better bass. On the software side, Samsung’s One UI is feature-rich and customizable, though some users might find the extra apps unnecessary.

Overall, the Galaxy S23 FE delivers strong performance, an excellent display, and dependable battery life, with minor trade-offs in heating and low-light camera performance. For its price, it feels like a well-balanced Android phone that gets most things right.
"""

class ComplexReview(TypedDict):
    summary : Annotated[str, "A concise summary of the review in one or two sentences."]
    sentiment : Annotated[Literal["POSITIVE", "NEGATIVE", "NEUTRAL"], "A sentiment of the review: positive, negative, or neutral."]
    pros : Annotated[Optional[list[str]], "A list of the positive aspects mentioned in the review."]
    cons : Annotated[Optional[list[str]], "A list of the negative aspects mentioned in the review."]
    key_themes : Annotated[list[str], "A list of the key themes of the products mentioned in the review."]

structured_chat_model = chat_model.with_structured_output(ComplexReview)
response = structured_chat_model.invoke(complex_review)
print(response)