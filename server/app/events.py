from flask import session
from flask_socketio import emit, join_room, leave_room, Namespace
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

# 클래스 기반 Namespace
class ChatNamepsace(Namespace):

    def on_connect(self):
        pass

    def on_disconnect(self):
        pass

    def on_text(self, data):
        train_data = pd.read_csv('C:\\Users\\ohzfl\\aichatbot_ktm\\trained_kt.csv')

        model = SentenceTransformer("C:\\Users\\ohzfl\\aichatbot_ktm\\output\\sts-2023-03-25_12-26-02")

        train_data['embedding'] = train_data.apply(lambda x : text2array(x.embedding), axis = 1)
        embedding = model.encode(data['msg'])
        train_data['score'] = train_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
        tem = train_data.loc[train_data['score'].idxmax()]
        room = session.get('room')
        emit('message', {'msg': "Q : " + data['msg'] + '\n' + tem['A']})

    def on_left(self, data):
        room = session.get('room')
        leave_room(room)
