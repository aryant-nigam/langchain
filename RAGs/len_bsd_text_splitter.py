from langchain_text_splitters import CharacterTextSplitter
import os
from langchain_community.document_loaders import PyPDFLoader
def test_character_text_splitter():
    text = """
    ucgagscgcj jjsibcs nsb cgus asbsjbc xajb ggbjb jbgba
    kni buohknbh bbba h jbbba jbb xabbh bojw nhuhb bib'
    hvuxiu hhih hiugs huhhus jkqi ios, hhhis hun ugwbb
    jbhs nihi nksqw jbqbqm bjb nbjqbs qgb gq dwwqd wwe
    """

    text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5, separator="")
    chunks = text_splitter.split_text(text)
    print(chunks)

def document_splitter():
   loader = PyPDFLoader("book.pdf")
   docs = loader.load()
   
   text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="")
   chunks = text_splitter.split_documents(docs)
   print(chunks[0])

if __name__ == "__main__":    # test_character_text_splitter()
    document_splitter()