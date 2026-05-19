from dotenv import load_dotenv
import os
import streamlit as st 
from langchain_core.prompts import PromptTemplate,load_prompt
from huggingface_hub import InferenceClient

load_dotenv()

st.header("Research Tool")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt("template.json")


prompt = template.invoke({"paper_input": paper_input, "style_input": style_input, "length_input": length_input})

if st.button("Summarize"):
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not token:
        st.error("HUGGINGFACEHUB_API_TOKEN is not set")
    else:
        client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=token)
        messages = [
            {"role": "system", "content": "You are a research assistant that summarizes papers clearly and accurately."},
            {"role": "user", "content": prompt.to_string()},
        ]

        result = client.chat_completion(messages, max_tokens=256)

        st.write(result.choices[0].message.content)