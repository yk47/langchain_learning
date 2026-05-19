from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv      
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnableLambda,RunnablePassthrough
import os

load_dotenv()

def word_count(text):
    return len(text.split())

model = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro", 
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),         
))

prompt = PromptTemplate(template="Write a joke about {text} ", input_variables=["text"])

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt , model , parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),  
    "word_count": RunnableLambda(word_count) #RunnableLambda allows you to wrap a simple function and use it as a runnable in your chain. Here, it is used to count the number of words in the generated joke.
})

final_chain = RunnableSequence(joke_gen_chain , parallel_chain)

result = final_chain.invoke({"text": "AI"})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)