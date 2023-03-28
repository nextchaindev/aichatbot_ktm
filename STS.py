from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datetime import datetime
import csv
import pandas as pd
import time
import numpy as np

def callback_f(score,epoch,steps):
    print(str(steps) + 'steps score : ' + str(score))

train_batch_size = 16
num_epochs = 500
model_save_path = 'output/sts'+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

model = SentenceTransformer('C:\\Users\\ohzfl\\aichatbot_ktm\\output', device='cuda')

train_samples = []
dev_samples = []

with open('data_sts.csv', newline='',encoding='UTF8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

        if row['split'] == 'dev':
            dev_samples.append(inp_example)
        else:
            train_samples.append(inp_example)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
start = time.time()
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          callback=callback_f,
          evaluation_steps=10,
          warmup_steps=warmup_steps,
          output_path=model_save_path)
end = time.time()
print(f"{end - start:.5f} sec")