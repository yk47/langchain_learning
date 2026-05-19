from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv      
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

import os

load_dotenv()

model = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro", 
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),         
))

parser = StrOutputParser()



class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(template="Classify the sentiment of the following feedback text into a positive or negative: \n {feedback} \n {format_instructions}", 
                         input_variables=["feedback"], partial_variables={"format_instructions": parser2.get_format_instructions()})

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(template="Write an appropriate response to this positive feedback: \n {feedback}", input_variables=["feedback"])
prompt3 = PromptTemplate(template="Write an appropriate response to this negative feedback: \n {feedback}", input_variables=["feedback"])

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment in the feedback")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "This is a wonderful smartphone."})

print("Final Output: ", result)

chain.get_graph().print_ascii()