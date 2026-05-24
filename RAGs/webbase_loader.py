from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.amazon.in/Cetaphil-Hydrating-Sulphate-Free-Niacinamide-Sensitive/dp/B01CCGW4OE/ref=zg_bs_c_beauty_d_sccl_2/523-8138064-4529163?pd_rd_w=FluWS&content-id=amzn1.sym.b908f532-cbe7-4274-8b24-b671acc58bd2&pf_rd_p=b908f532-cbe7-4274-8b24-b671acc58bd2&pf_rd_r=WCSSZMTTPPEJ6RPTFM1B&pd_rd_wg=PaoG4&pd_rd_r=5e3620a6-b199-48e9-9380-3ab11e93c391&pd_rd_i=B01CCGW4OE&th=1")

description = loader.load()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
summary_prompt = PromptTemplate(template="Summarize the following product description: {description}. Mention the composition if available, product_reviews if available, price if available", input_variables=["description"])

summary_chain = summary_prompt | model | StrOutputParser()

res = summary_chain.invoke({"description": description})
print(res)