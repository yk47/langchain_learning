from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv      
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
import os

load_dotenv()

model = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro", 
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),         
))

prompt = PromptTemplate(template="Write a summary of the following poem - \n {poem}", input_variables=["poem"])

parser = StrOutputParser()

# TextLoader is a simple document loader that loads text files. It takes in the file path and encoding as input and returns a list of Document objects. Each Document object contains the page content and metadata of the loaded text file.
loader = TextLoader("cricket.txt", encoding="utf-8")
docs = loader.load()

print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({"poem": docs[0].page_content})

print("Final Output: ", result)