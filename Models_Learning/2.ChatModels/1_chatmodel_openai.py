from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0.7, max_completions_tokens=1024)

result = model.invoke("What is the capital of India?")

print(result.content)
