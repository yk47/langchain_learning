from langchain_community.document_loaders import WebBaseLoader



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

prompt = PromptTemplate(template="Answer the following question - \n {question} from the following text: \n {text}", input_variables=["question", "text"])

parser = StrOutputParser()
url = "https://en.wikipedia.org/wiki/MacBook_Air"

# WebBaseLoader is a document loader that loads web pages. It takes in the URL of the web page as input and returns a list of Document objects. Each Document object contains the page content and metadata of the loaded web page.
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({"question": "What is the product that we are talking about?", "text": docs[0].page_content})

print("Final Output: ", result)