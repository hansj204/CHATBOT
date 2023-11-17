import pandas as pd
import re
import tensorflow_datasets as tfds
import tensorflow as tf
import os
from .classes import *
from .transformer import transformer

current_directory = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(current_directory, 'data')

train_file_path = os.path.join(data_folder_path, 'talk_training_data.csv')
model_weights_path = os.path.join(data_folder_path, 'talk_model_weights.h5')

# 1. 데이터 로드하기
train_data = pd.read_csv(train_file_path)
train_data.head()

# 2. 단어 집합 생성
questions = [] 
answers = []
MAX_LENGTH = 40

for sentence in train_data['Q']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    questions.append(sentence)
    

for sentence in train_data['A']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    answers.append(sentence)
    
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2**13)

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

VOCAB_SIZE = tokenizer.vocab_size + 2

# 3. 트랜스포머 생성 & 가중치 로드
D_MODEL = 256 #512
NUM_LAYERS = 2 #6
NUM_HEADS = 8
DFF = 512 #2048
DROPOUT = 0.1

def transformer_model():
    tf.keras.backend.clear_session()

    return transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        dff=DFF,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)