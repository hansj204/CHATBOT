from django.shortcuts import render
from .models import predict
import json
from django.http import JsonResponse

def index(request):
    return render(request, 'index.html')

def chat(request):
    if request.method == 'GET':
        return render(request, 'chat.html')
    
    elif request.method == 'POST':
        
        data = json.loads(request.body.decode('utf-8'))
        sentence = data.get('sentence')
        
        predicted_sentence = predict(sentence)

        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))
        
        return render(request, 'chat.html', JsonResponse({'predicted_sentence': predicted_sentence}))

def result(request):
    mbti_type = 'INTJ'
    
    return render(request, 'result.html', {'mbti_type': mbti_type})
