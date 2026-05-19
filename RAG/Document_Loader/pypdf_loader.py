from langchain_community.document_loaders import PyPDFLoader

# PyPDFLoader is a document loader that loads PDF files. It takes in the file path as input and returns a list of Document objects. Each Document object contains the page content and metadata of the loaded PDF file.
loader = PyPDFLoader("Flutter_Git_Basics_Detailed_Docs.pdf")
docs = loader.load()
print(len(docs))
print(docs[0].page_content)
print(docs[1].metadata)