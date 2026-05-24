from langchain_community.document_loaders import PyPDFLoader
import os

loader = PyPDFLoader("/Users/aryantnigam/Downloads/my docs/formulo_personal_data/AryantSrSoftwareEngineerOfferLetter.pdf")
print(loader.load())