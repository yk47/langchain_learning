from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient, login

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not token:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is not set")

login(token=token, add_to_git_credential=False)

client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct")
messages = [{"role": "user", "content": "What is the capital of India?"}]

response = client.chat_completion(messages, max_tokens=20)
print(response.choices[0].message.content)