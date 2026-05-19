from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

# RecursiveCharacterTextSplitter is a text splitter that splits text based on a specified separator character. It takes in the chunk size, chunk overlap, and separator as input and returns a list of Document objects for the split text. Each Document object contains the page content and metadata of the split text.
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])
    