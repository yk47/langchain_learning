from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader

# DirectoryLoader is a document loader that loads documents from a directory. It takes in the path to the directory, a glob pattern to match the files, and the loader class to use for loading the files. It returns a list of Document objects for all the files that match the glob pattern in the specified directory.
loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)
   