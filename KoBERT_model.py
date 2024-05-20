import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import gluonnlp as nlp
import numpy as np


from BERTClassifier import *
from BERTSentenceTransform import *
from BERTDataset import *
from get_kobert_model import *
from kobert_tokenizer import *


model = None
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel, vocab = get_kobert_model('skt/kobert-base-v1',tokenizer.vocab_file)  # calling the bert model and the vocabulary
#device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device('cpu')

model = BERTClassifier(bertmodel,  dr_rate=0.4).to(device)


def load_model():
    global model
    model = BERTClassifier(bertmodel,  dr_rate=0.4).to(device)

    model.load_state_dict(torch.load('train_5epch.pt', map_location=device), strict = False)
    model.eval()
    #print(model)

def load_dataset(predict_sentence):
    #tokenizer = get_tokenizer()   
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    data = [predict_sentence, '0']
    dataset_another = [data]
    another_test = BERTDataset(dataset_another, 0, 1,tokenizer,vocab, max_len=64, pad=True, pair=False)
    return DataLoader(another_test, batch_size = 32, num_workers = 2) # torch 형식 변환

def inference(predict_sentence): # input = 보이스피싱 탐지하고자 하는 sentence
    print("※ KoBERT 추론 시작 ※")

    model.load_state_dict(torch.load('train_5epch.pt', map_location=device), strict = False)
    model.eval()

    test_dataloader = load_dataset(predict_sentence)

    print(predict_sentence)

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        #print("valid_length : ",valid_length)
        #print("token_ids : ", token_ids)
        #print("segment_ids : ", segment_ids)

        #print('***model*** : ', model)

        out = model(token_ids, valid_length, segment_ids)
        #print("outcome :", out)

        softm_probabilities = F.softmax(out, dim=1)
        #print("softmax probabilities : ",softm_probabilities)
        _, predicted_class = torch.max(softm_probabilities, 1)
        softm_confidence_level = softm_probabilities[0][predicted_class.item()].item() * 100
        #print("predict : ", predicted_class)
        #print("softmax 확률 : ",softm_confidence_level)
        #print(softm_probabilities[0][1].item()*100)


        sigm_probabilities = F.sigmoid(out)
        #print("sigmoid probabilities : ", sigm_probabilities)
        sigm_confidence_level = sigm_probabilities[0][predicted_class.item()].item() * 100
        #print("sigmoid 확률 : ",sigm_confidence_level)
        print(sigm_probabilities[0][1].item()*100)


        result = False
        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("일반 음성 전화")
            elif np.argmax(logits) == 1:
                test_eval.append("보이스피싱 전화")
                result = True

        print("▶ 입력하신 내용은 '" + test_eval[0] + "' 입니다.")
        return result

def run(text):
    
    load_model()
    return inference(text)

def inf(text):
    return inference(text)

