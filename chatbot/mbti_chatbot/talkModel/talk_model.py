import re
import os
import pickle
import tensorflow as tf

from nltk.translate.bleu_score import sentence_bleu
from PyKakao import KoGPT
from decouple import config

from .classes import *
from .transformer import transformer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

current_directory = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(current_directory, 'data')

tokenizer_path = os.path.join(data_folder_path, 'tokenizer.pkl')
model_weights_path = os.path.join(data_folder_path, 'weights_personas_5.h5')

KoGPT_KEY = config('KoGPT_KEY')
api = KoGPT(service_key = KoGPT_KEY)

with open(tokenizer_path, 'rb') as f:
    talk_tokenizer = pickle.load(f)
    
tf.keras.backend.clear_session()

START_TOKEN, END_TOKEN = [talk_tokenizer.vocab_size], [talk_tokenizer.vocab_size + 1]
VOCAB_SIZE = talk_tokenizer.vocab_size + 2
MAX_LENGTH = 100

talk_model = transformer(vocab_size=VOCAB_SIZE, num_layers=2, dff=512, d_model=256, num_heads=8, dropout=0.1)
talk_model.load_weights(model_weights_path)    
    
def preprocess_user_msg(msg):
    msg = re.sub('[^가-힣a-zA-Z0-9]', ' ', msg)
    msg = re.sub(r'[^\w\s]', '', ' '.join(msg.split()))  
    msg = re.sub(r'\([^)]*\)', '', msg)
    msg = msg.strip()
    return msg

def preprocess_bot_msg(msg):
    msg = preprocess_user_msg(msg)
       
    index_of_q = msg.find('Q')
    endings = ['!', '.', ',']
    
    if index_of_q != -1:
        msg = msg[:index_of_q]
        
    for ending in endings:
        index = msg.find(ending)
    
        if index != -1:
            msg = msg[:index + 1]
            break
        
    return msg

def calculate_bleu_score(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    return sentence_bleu(reference, candidate)

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

def is_valid_msg(user_msg, bot_msg):
    print(f"bot_msg: {bot_msg}")
    print(calculate_bleu_score(bot_msg, user_msg))
    return calculate_bleu_score(bot_msg, user_msg) >= 0.7 and (0 < len(bot_msg) <= MAX_LENGTH)

def ask_chatbot(user_msg):
    bot_msg = get_bot_msg(user_msg)
    
    if not is_valid_msg(user_msg, bot_msg):
        user_msg = '나는' + user_msg
        bot_msg = get_bot_msg('나는' + user_msg)
        
        if not is_valid_msg(user_msg, bot_msg): return ''
        
    return bot_msg

def ask_gpt(question, user_msg):
    text = '''정보: 말투 친절함, 익명, 한국인
    정보를 바탕으로 Q의 문장에 공감하며 존댓말로 답장하세요. 단, 의문문으로 답하지 마세요.
    Q: 안녕하세요.
    A: ''' + question + '''
    Q: 나는 ''' + user_msg + '''
    A: '''
    
    attempts = 0
    max_attempts = 3
    response = ''

    while attempts < max_attempts:
        response = api.generate(text, 40, temperature=0.3, top_p=0.85)
        response = response['generations'][0]['text']
        response = preprocess_bot_msg(response)
                
        if calculate_bleu_score(response, user_msg) >= 0.7 and (0 < len(response) <= 40): break
        
        attempts += 1

    if attempts > max_attempts:
        response = "네, 그렇군요. 다음 주제로 이야기해볼까요?"

    return response
