from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
import streamlit as st
import torch
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.schema import HumanMessage, SystemMessage
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

device = "cuda:0" if torch.cuda.is_available() else "cpu"

llm = HuggingFaceEndpoint(
    task="text-generation",
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    max_new_tokens=512,
    top_k= 10,
    top_p= 0.9,
    temperature= 0.1,
    repetition_penalty=1.03,
    huggingfacehub_api_token= HUGGINGFACEHUB_API_TOKEN
)

model = ChatHuggingFace(llm = llm)

# print(model.model_id)

st.header("Chat with your Data")
file = st.file_uploader("Upload your File here", type="csv")
if file is not None:
    path_to_save = "data.csv"
    with open(path_to_save, "wb") as f:
        f.write(file.getvalue())
    print("File loaded!!!!!!")
    df = pd.read_csv("data.csv")
    agent = create_pandas_dataframe_agent(model, df, verbose=True)
    query = st.text_area("Enter your query", height=200)
    if st.button("Run Query"):
        if len(query) > 0:
            st.info(f"Your query: {query}")
            messages = [
                SystemMessage(
                            content='''You're an expert and a helpful assistant. 
                                        You have expertise in answering the questions related to Data Science.
                                        You always provide the complete answers to the user question with proper explainations.
                                        You always append the results to the output section of the result'''
                              ),
                HumanMessage(
                    content=f'execute the following query: {query}, and append the answer in the ouput section of the result.'
                ),
            ]
            res = agent(messages)
            print(res)
            st.write(res)















