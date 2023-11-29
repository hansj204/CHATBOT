import pandas as pd
import re
import tensorflow_datasets as tfds
import tensorflow as tf
import os
from .classes import *
from .transformer import transformer
import pickle

current_directory = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(current_directory, 'data')

tokenizer_path = os.path.join(data_folder_path, 'tokenizer.pkl')
model_weights_path = os.path.join(data_folder_path, 'talk_model_weights.h5')

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
    
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2
MAX_LENGTH = 40

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