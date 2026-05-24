from langchain_text_splitters import RecursiveCharacterTextSplitter
text = """
In AI, tokens help handle unknown words better by breaking them into smaller parts. If a model only stored full words, it might fail when it sees a new word it has never learned before. With tokens, a word can be split into pieces like prefixes, roots, or suffixes. Even if the complete word is new, the AI can understand its parts and process the meaning.

Tokens also reduce vocabulary size. Instead of storing millions of full words, the model stores a smaller set of reusable tokens. These tokens can combine to form many different words, making the system more efficient and flexible for language processing.
"""

chuncks = RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=5,
).split_text(text)

print(chuncks)


