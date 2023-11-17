from django.shortcuts import render
from .models import predict
from django.http import JsonResponse
import json

def index(request):
    return render(request, 'index.html')

def chat(request):
    if request.method == 'GET':
        return render(request, 'chat.html')
    
    elif request.method == 'POST':
        
        data = json.loads(request.body.decode('utf-8'))
        sentence = data.get('sentence')
        
        predicted_sentence = predict(sentence)

        return JsonResponse({'predicted_sentence': predicted_sentence})

def result(request):
    mbti_type = 'INTJ'
    
    return render(request, 'result.html', {'mbti_type': mbti_type})
