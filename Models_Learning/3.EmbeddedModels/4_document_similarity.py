from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting style and leadership skills. He has been the captain of the Indian national team and has achieved numerous records in international cricket",
    "MS Dhoni is a former Indian cricketer and captain of the Indian national team. He is known for his calm demeanor, exceptional wicket-keeping skills, and finishing abilities in limited-overs cricket.",
    "Sachin Tendulkar is a former Indian cricketer widely regarded as one of the greatest batsmen in the history of cricket. He holds numerous records, including being the highest run-scorer in both Test and One Day International formats.",
    "Rohit Sharma is an Indian cricketer known for his elegant batting style and ability to score big centuries. He has been a key player for the Indian national team and has achieved several records, including the highest individual score in One Day Internationals."
    "Jaspreet Bumrah is an Indian cricketer known for his unique bowling action and ability to bowl yorkers consistently. He has been a crucial part of the Indian national team's bowling attack and has achieved success in all formats of the game."
]

query = "tell me about Virat Kohli"

document_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

similarity_scores = cosine_similarity([query_embedding], document_embeddings)[0]
index, score =  sorted(list(enumerate(similarity_scores)),key=lambda x: x[1],)[::-1][:3]
print(query)
print(documents[index])
print(" Similarity Score is : ", score)