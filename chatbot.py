import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import urllib.request
from sentence_transformers import SentenceTransformer

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
    if(score >= 0.9):
        return [question,tem['A'],score]
    else:
        return [question,"제가 질문을 이해하지 못했어요",score]


train_data = pd.read_csv('trained.csv')

#model = SentenceTransformer("input your directory")
model = SentenceTransformer("C:\\Users\\ohzfl\\aichatbot_ktm")

train_data['embedding'] = train_data.apply(lambda x : text2array(x.embedding), axis = 1)

print(return_answer("나는 병원에서 라면을 읽었어"))
print(return_answer("집에 가고 싶어"))
print(return_answer("라면 먹을 거야"))
print(return_answer("결혼하고 싶어"))
print(return_answer("감기에 걸렸어"))
print(return_answer(" "))
print(return_answer("사랑해"))
print(return_answer("너는 누구야?"))
print(return_answer("너는 누구니?"))
print(return_answer("삼각함수"))
print(return_answer("asdf"))
print(return_answer("어제 나가사키 짬뽕을 사 먹었는데, 아시다시피 나가사키는 원폭이 떨어진 도시 중 하나죠. 핵전쟁이 참 무섭습니다."))
print(return_answer("그래 나도 잘 치료되서 원만히 사회생활하며 동네사람들이 다 날 좋아해 결국 말한다는 건 시간버리지말고 달빛도 보면서 잠 안되겠으면"))
print(return_answer("나는 점심 때 수와 박을 먹었다."))
print(return_answer("남자와 여자 목소리로 환청과 환시가 들린다. 어떤 사이트에서 나에 대해 민원을 넣어서 그런 것 같다."))
print(return_answer("화해하자사스케 난 너가 세상에서 제일 존경스럽고 미친듯이 아름다워 머라고 하든 어깨피고다녀 개병신같은새끼들이 인스타게시물에 머올리는거냐고 지랄하든 말든 사스케는 이타치의 동생이 아니다"))
print(return_answer("자기들은 시켜서 그랬다고 하며 자꾸 담배를 끊으라고 한다. 목욕하다 항문에서 칩(chip) 같은 것이 나와 제거했다. 목에도 남아 있는데 찾을 수 없다. 주로 삐 소리가 나고 하품하면 강해진다. 코에서 물방울이 터지는 것 같고 항문도 울퉁불퉁해졌다."))
print(return_answer("너는 라일락하일락 나는 통카닥콩카닥이 부루루르르르 요기조기 싶다싶다 여기 조기 뽜지직꽈지직하는 대장간날개 쉬이이잉 사슬이 너무 오도로토토옹해서 푹자고 싶다가 오로로로롱하다. 왜 옹알이 알아듣다가 맛있게 먹었습니다 고마워요 대들보님 통카닥콩카닥 못하다가 너는 라일락하일락 아니다가 오토토토통만하다. 야로빠가. 우리 배가 꼬리르르락 위장 넣어줘. 라일락하일락"))
print(return_answer("대학원 가는거니?"))
print(return_answer("NLP가 뭔데"))
print(return_answer("I am hungry"))
print(return_answer("What time is now?"))
print(return_answer("I hate homework"))