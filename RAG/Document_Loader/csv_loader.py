from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import data

# CSVLoader is a document loader that loads CSV files. It takes in the file path and encoding as input and returns a list of Document objects. Each Document object contains the page content and metadata of the loaded CSV file.
loader = CSVLoader(file_path="Social_Network_Ads.csv", encoding="utf-8")

docs = loader.load()

print(len(docs))

print(docs[1])