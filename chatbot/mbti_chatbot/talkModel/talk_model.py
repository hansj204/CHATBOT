import re
import os
import pickle
import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyKakao import KoGPT
from decouple import config

from .classes import *
from .transformer import transformer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

current_directory = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(current_directory, 'data')

tokenizer_path = os.path.join(data_folder_path, 'tokenizer.pkl')
model_weights_path = os.path.join(data_folder_path, 'weights_personas.h5')

KoGPT_KEY = config('KoGPT_KEY')
api = KoGPT(service_key = KoGPT_KEY)

with open(tokenizer_path, 'rb') as f:
    talk_tokenizer = pickle.load(f)
    
tf.keras.backend.clear_session()

START_TOKEN, END_TOKEN = [talk_tokenizer.vocab_size], [talk_tokenizer.vocab_size + 1]
VOCAB_SIZE = talk_tokenizer.vocab_size + 2
MAX_LENGTH = 40

talk_model = transformer(vocab_size=VOCAB_SIZE, num_layers=2, dff=512, d_model=256, num_heads=8, dropout=0.1)
talk_model.load_weights(model_weights_path)    
    
def preprocess_user_msg(msg):
    msg = re.sub('[^가-힣a-zA-Z0-9]', ' ', msg)
    msg = re.sub(r'[^\w\s]', '', ' '.join(msg.split()))  
    msg = re.sub(r'\([^)]*\)', '', msg)
    msg = msg.strip()
    return msg

def preprocess_bot_msg(msg):
    if 0 == len(msg): return msg
    
    last_punctuation_index = max(msg.rfind('.'), msg.rfind(','), msg.rfind('!'))
    msg = msg[:last_punctuation_index + 1]
        
    return msg

def is_Right_Answer(user_msg, bot_msg):
    if not (0 < len(bot_msg) <= MAX_LENGTH or "?" != bot_msg[-1]): return False

    vectorizer = CountVectorizer().fit_transform([user_msg, bot_msg])
    cosine_sim = cosine_similarity(vectorizer)

    return cosine_sim[0][1] >= float(0.6)

def evaluate(user_msg):
  sentence = preprocess_user_msg(user_msg)
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

def get_bot_msg(user_msg):
    prediction = evaluate(user_msg)
    bot_msg = talk_tokenizer.decode([i for i in prediction if i < talk_tokenizer.vocab_size])
    return preprocess_bot_msg(bot_msg)

def ask_chatbot(question, user_msg):
    bot_msg = get_bot_msg(user_msg)
        
    if True == is_Right_Answer(user_msg, bot_msg): 
        return bot_msg
    else:
        return ask_gpt(question, user_msg)

def ask_gpt(question, user_msg):
    text = '''정보: 말투 친절함, 익명, 한국인
    정보를 바탕으로 Q의 문장에 공감하며 존댓말로 답장하세요. 단, 질문에 대해 공감형 답변을 하세요.
    Q: 안녕하세요.
    A: ''' + question + '''
    Q: 나는 ''' + user_msg + '''
    A: '''
    
    attempts = 0
    max_attempts = 2
    response = ''

    while attempts < max_attempts:
        response = api.generate(text, MAX_LENGTH, temperature=0.3, top_p=0.85)
        response = response['generations'][0]['text']
        response = preprocess_bot_msg(response)
                
        if 0 < len(response) and "?" != response[-1] : return response
        
        attempts+= 1
                
    return response