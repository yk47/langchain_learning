from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pathlib import Path

# chat template
chat_template = ChatPromptTemplate([
    ('system', "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}'),
])

chat_history = []

# load chat history
history_file = Path(__file__).parent / "chat_history.txt"
if history_file.exists():
    with open(history_file) as f:
        chat_history.extend(f.readlines())

print(chat_history)

# create prompt
prompt = chat_template.invoke({"chat_history": chat_history, "query": "What is the status of my order?"})
print(prompt)