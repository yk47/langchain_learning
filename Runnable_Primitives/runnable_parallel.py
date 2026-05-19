from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv      
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence

import os

load_dotenv()

model1 = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"), 
))
model2 = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro", 
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),         
))

prompt1 = PromptTemplate(template="Generate a tweet about {topic} ", input_variables=["topic"])
prompt2 = PromptTemplate(template="Generate a LinkedIn post about {topic} ", input_variables=["topic"])

parser = StrOutputParser()  

# RunnableParallel executes the provided runnables in parallel and returns a dictionary with the outputs.
parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1 , model1 , parser),
    "linkedin": RunnableSequence(prompt2 , model2 , parser)
})

result = parallel_chain.invoke({"topic": "AI"})

print(result['tweet'])
print(result['linkedin'])