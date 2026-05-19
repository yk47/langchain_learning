from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv      
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
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

joke_gen_chain = RunnableSequence(prompt1 , model ,parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(), #RunnablePassthrough simply passes the input to the output without any modifications. Here it is used to pass the generated joke from joke_gen_chain to the parallel_chain.
    "explanation": RunnableSequence(prompt2 , model , parser)
})

final_chain = RunnableSequence(joke_gen_chain , parallel_chain)

result = final_chain.invoke({"text": "cricket"})

print("Joke: ", result['joke'])
print("Explanation: ", result['explanation'])