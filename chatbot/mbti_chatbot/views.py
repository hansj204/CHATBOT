from django.shortcuts import render
# from models import evaluate, tokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def index(request):
    return render(request, 'index.html')

def chat(request):
    return render(request, 'chat.html')

def result(request):
    print(request)
    return render(request, 'result.html')

# def predict(sentence):
#     prediction = evaluate(sentence)
#     predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])

#     print('Input: {}'.format(sentence))
#     print('Output: {}'.format(predicted_sentence))

#     return predicted_sentence
