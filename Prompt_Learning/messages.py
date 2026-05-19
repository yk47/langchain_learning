from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not token:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is not set")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=token,
    max_new_tokens=256,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful assistant that provides concise and accurate answers."),
    HumanMessage(content="Tell me about LangChain."),
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)