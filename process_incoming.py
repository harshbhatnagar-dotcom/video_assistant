import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")

groq_url="https://api.groq.com/openai/v1"

groq=OpenAI(base_url=groq_url,api_key=groq_api_key)
MODEL="openai/gpt-oss-120b"

df=joblib.load("embeddings.joblib")

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "qwen3-embedding:0.6b",
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

system_prompt = """
You are an AI teaching assistant for a course. You have access to video chunks data containing:

- video title
- video number
- start time (seconds)
- end time (seconds)
- text of the chunk

Your job is to answer user questions **only related to the course content**. 
When answering:

1. Identify which video(s) contain the relevant content.
2. Specify the **timestamps** in minutes (start time only not the end time) where the information is taught.
3. Guide the user to the exact video and timestamp for reference.
4. If the question is unrelated to the course, politely tell the user that you can only answer questions related to the course.
"""

def question_answer(user_input,history):
    question_embedding=create_embedding([user_input])[0]
    
    similarities=cosine_similarity(np.vstack(df["embedding"]),[question_embedding]).flatten()
    top_results=5
    max_indx=similarities.argsort()[::-1][0:top_results]
    new_df=df.loc[max_indx]
    user_prompt = f"""
Here are the video chunks (video title, number, start time, end time, and text):

{new_df[["title","number","start","end","text"]].to_json(orient='records')}

User question: {user_input}
According to the user input provide where and how much content is taught in which video(in which video at what timestamp dont provide the end time)
"""

    messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}]

    response=groq.chat.completions.create(model=MODEL,messages=messages)
    return response.choices[0].message.content

