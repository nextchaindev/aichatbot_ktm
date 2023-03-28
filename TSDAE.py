from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader
import nltk
import time

def callback_f(score,epoch,step)->None:
    print(str(epoch) + 'steps\' score: ' + str(score))

epoch = 350

model_name = 'C:\\Users\\ohzfl\\aichatbot_ktm\\ko-sroberta-multitask'
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cuda')

with open('C:\\Users\\ohzfl\\aichatbot_ktm\\data_tsdae.txt','r',encoding='UTF8') as fi:
    train_sentences = fi.read().split('\n')

train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)
start = time.time()
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=epoch,
    weight_decay=0,
    scheduler='constantlr',
    optimizer_params={'lr': 3e-5},
    callback=callback_f,
    show_progress_bar=True
)
end = time.time()
model.save('C:\\Users\\ohzfl\\aichatbot_ktm\\output')
print(f"{end - start:.5f} sec")