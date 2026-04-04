import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "qwen3-embedding:0.6b",
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding


jsons = os.listdir("jsons")  # List all the jsons 
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")

    texts = []

    for c in content['chunks']:
        texts.append(c['text'])

    embeddings = create_embedding(texts)
       
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk) 

 



df = pd.DataFrame.from_records(my_dicts)

# Save this data frame
joblib.dump(df,"embeddings.joblib")



