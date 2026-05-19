from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Flutter_Git_Basics_Detailed_Docs.pdf")
docs = loader.load()

# CharacterTextSplitter is a text splitter that splits text based on a specified separator character. It takes in the chunk size, chunk overlap, and separator as input and returns a list of Document objects for the split text. Each Document object contains the page content and metadata of the split text.
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=15,separator='')

chunks = splitter.split_documents(docs)
print(chunks[0].page_content)
print(chunks[0].metadata)
