import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import urllib.request
from sentence_transformers import SentenceTransformer

#urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
train_data = pd.read_csv('ktdata.csv')
print(train_data.head())
model = SentenceTransformer("C:\\Users\\ohzfl\\aichatbot_ktm\\output\\sts-2023-03-18_19-21-44")

train_data['embedding'] = train_data.apply(lambda row: model.encode(row.Q), axis = 1)

train_data.to_csv("trained_kt.csv")