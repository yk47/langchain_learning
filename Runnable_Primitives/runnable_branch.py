from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv      
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough,RunnableBranch
import os

load_dotenv()



model = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro", 
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),         
))

prompt1 = PromptTemplate(template="Write a detailed report on {topic}", input_variables=["topic"])

prompt2 = PromptTemplate(template="Summarize the following text - {text}", input_variables=["text"])

parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser

#RunnableBranch allows you to create branches in your chain based on a condition. Here, it checks if the generated report has more than 500 words, and if so, it summarizes the report using another prompt and model.
    
branch_chain = RunnableBranch(
    (lambda x: len(x.split())>100, prompt2 | model | parser), #Condition and the chain to execute if the condition is true.,
    RunnablePassthrough()
)

final_chain = report_gen_chain | branch_chain

result = final_chain.invoke({"topic": "Russia vs Ukraine "})

print( result)