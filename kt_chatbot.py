import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import urllib.request
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

def text2array(text):
    tem = text[1:-1].split()
    tem = list(map(float,tem))
    return np.array(tem)

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def return_answer(question):
    embedding = model.encode(question)
    train_data['score'] = train_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
    tem = train_data.loc[train_data['score'].idxmax()]
    score = tem['score']
    return train_data.nlargest(3,'score', keep = 'all').drop(labels=['A','label','embedding'],axis=1)
    #if(score >= 0.8):
    #    return [question,tem['A'],score]
    #else:
    #    return [question,"제가 질문을 이해하지 못했어요",tem['A'],score]


train_data = pd.read_csv('trained_kt.csv')

#model = SentenceTransformer("input your directory")
model = SentenceTransformer("C:\\Users\\ohzfl\\aichatbot_ktm\\output\\sts-2023-03-25_12-26-02")

train_data['embedding'] = train_data.apply(lambda x : text2array(x.embedding), axis = 1)

while(True):
    x = input("무엇을 도와드릴까요?")
    if(x == "q"):
        break
    else:
        print(return_answer(x))