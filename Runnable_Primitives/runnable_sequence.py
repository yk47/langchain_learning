from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv      
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os

load_dotenv()

model = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro", 
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),         
))

prompt1 = PromptTemplate(template="Write a joke about {text} ", input_variables=["text"])

parser = StrOutputParser()

prompt2 = PromptTemplate(template="Explain the following joke - {text}", input_variables=["text"])

# RunnableSequence executes the provided runnables in sequence, passing the output of one as the input to the next. 
chain = RunnableSequence(prompt1 , model ,parser,prompt2 , model , parser)

print(chain.invoke({"text": "AI"}))