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

tf.keras.backend.clear_session()

talk_model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)
    
talk_model.load_weights(model_weights_path)

# 사용자 메세지 전처리
def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

# 응답 메세지에 해당하는 단어 예측
def evaluate(sentence):
    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = talk_model(inputs=[sentence, output], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        output = tf.concat([output, predicted_id], axis=-1)
        
    return tf.squeeze(output, axis=0)