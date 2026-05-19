from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()


class Facts(BaseModel):
    fact1: str = Field(description="Fact 1 about the topic")
    fact2: str = Field(description="Fact 2 about the topic")
    fact3: str = Field(description="Fact 3 about the topic")
    fact4: str = Field(description="Fact 4 about the topic")
    fact5: str = Field(description="Fact 5 about the topic")


parser = PydanticOutputParser(pydantic_object=Facts)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template="""
Give me 5 facts about {topic}

{format_instructions}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

prompt = template.format_prompt(topic="Black Hole")

result = model.invoke(prompt.to_messages())

final_result = parser.parse(result.content)

print(final_result)