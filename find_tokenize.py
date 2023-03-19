import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import urllib.request
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("C:\\Users\\ohzfl\\aichatbot_ktm\\ko-sroberta-multitask")

train_data = pd.read_csv('ktdata.csv')

#for i in train_data['Q']:
#    print(model.tokenizer.tokenize(i))    

while(True):
    x = input("무엇을 도와드릴까요?")
    if(x == "q"):
        break
    else:
        print(model.tokenizer.tokenize(x))