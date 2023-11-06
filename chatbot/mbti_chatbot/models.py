# from django.db import models
# from django.contrib.auth.models import User
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import re
# import urllib.request
# import time
# import tensorflow_datasets as tfds
# import tensorflow as tf

# urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
# train_data = pd.read_csv('ChatBotData.csv')
# train_data.head()

# questions = [], answers = [], MAX_LENGTH = 40

# for sentence in train_data['Q']:
#     sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
#     sentence = sentence.strip()
#     questions.append(sentence)
    

# for sentence in train_data['A']:
#     sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
#     sentence = sentence.strip()
#     answers.append(sentence)
    
# tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2**13)

# START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# VOCAB_SIZE = tokenizer.vocab_size + 2

# def tokenize_and_filter(inputs, outputs):
#     tokenized_inputs, tokenized_outputs = [], []

#     for (sentence1, sentence2) in zip(inputs, outputs):
#         sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
#         sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

#         tokenized_inputs.append(sentence1)
#         tokenized_outputs.append(sentence2)

#     # 패딩
#     tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
#         tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
#     tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
#         tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

#     return tokenized_inputs, tokenized_outputs

# questions, answers = tokenize_and_filter(questions, answers)

# BATCH_SIZE = 64
# BUFFER_SIZE = 20000

# dataset = tf.data.Dataset.from_tensor_slices((
#     {
#         'inputs': questions,
#         'dec_inputs': answers[:, :-1] 
#     },
#     {
#         'outputs': answers[:, 1:]
#     },
# ))

# dataset = dataset.cache()
# dataset = dataset.shuffle(BUFFER_SIZE)
# dataset = dataset.batch(BATCH_SIZE)
# dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# def preprocess_sentence(sentence):
#   sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
#   sentence = sentence.strip()
#   return sentence

# def evaluate(sentence):
#     sentence = preprocess_sentence(sentence)
#     sentence = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)
#     output = tf.expand_dims(START_TOKEN, 0)

#     for i in range(MAX_LENGTH):
#         predictions = model(inputs=[sentence, output], training=False)

#         predictions = predictions[:, -1:, :]
#         predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

#         if tf.equal(predicted_id, END_TOKEN[0]):
#             break

#     output = tf.concat([output, predicted_id], axis=-1)

#     return tf.squeeze(output, axis=0)