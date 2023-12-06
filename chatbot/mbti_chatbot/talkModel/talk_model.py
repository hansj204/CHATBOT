import re
import os
import json
import pickle
import tensorflow as tf

from konlpy.tag import Okt
from keybert import KeyBERT
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.similarities import Similarity
from PyKakao import KoGPT

from .classes import *
from .transformer import transformer

okt = Okt()

current_directory = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(current_directory, 'data')

tokenizer_path = os.path.join(data_folder_path, 'tokenizer.pkl')
model_weights_path = os.path.join(data_folder_path, 'weights_7.h5')

with open(tokenizer_path, 'rb') as f:
    talk_tokenizer = pickle.load(f)
    
tf.keras.backend.clear_session()

START_TOKEN, END_TOKEN = [talk_tokenizer.vocab_size], [talk_tokenizer.vocab_size + 1]
VOCAB_SIZE = talk_tokenizer.vocab_size + 2
MAX_LENGTH = 100

D_MODEL = 256 #512
NUM_LAYERS = 2 #6
NUM_HEADS = 8
DFF = 512 #2048
DROPOUT = 0.1

talk_model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)
    
talk_model.load_weights(model_weights_path)    
    
def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

def extract_keywords_bert_from_string(text):
    kw_extractor = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = kw_extractor.extract_keywords(text)
    meaningful_keywords = [(kw, weight) for kw, weight in keywords if weight >= 0.5]
    return [item[0] for item in meaningful_keywords][:5]

def evaluate(question, user_msg):
  input_text = question + ' ' + user_msg
  sentence = preprocess_sentence(input_text)
  sentence = tf.expand_dims(START_TOKEN + talk_tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = talk_model(inputs=[sentence, output], training=False)

    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    if tf.equal(predicted_id, END_TOKEN[0]):
      break
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)

def calculate_topic_similarity(user_msg, bot_msg):
    user_tokens = extract_keywords_bert_from_string(user_msg)
    bot_tokens = extract_keywords_bert_from_string(bot_msg)

    dictionary = Dictionary([user_tokens, bot_tokens])

    corpus = [dictionary.doc2bow(tokens) for tokens in [user_tokens, bot_tokens]]

    lda_model = LdaModel(corpus, id2word=dictionary, num_topics=2)

    user_topic = lda_model[dictionary.doc2bow(user_tokens)]
    bot_topic = lda_model[dictionary.doc2bow(bot_tokens)]

    similarity_index = Similarity('', [user_topic], num_features=len(dictionary))
    similarity_score = similarity_index[bot_topic]

    return similarity_score[0]

api = KoGPT(service_key = "1e9517c7b61c42c24ba5f0d684d5922c")

def ask_gpt(question, user_msg):
    text = '''정보: 말투 친절함, 10자 이상의 감탄문
    정보를 바탕으로 으로 질문에 답하세요. 
    Q: ''' + question + '''
    A: ''' + user_msg
    
    response_json = api.generate(text, 32, temperature=0.3, top_p=0.85)
    response = json.loads(response_json)

    return response['generations'][0]['text']
