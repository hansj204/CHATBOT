from django.shortcuts import render, redirect
from .models import predict , questionGet ,QuestionDB
from django.http import JsonResponse
import json

def index(request):
    request.session['chat_history'] = []
    request.session['chat_questionCheck'] = "NO"
    QuestionDB.objects.all().delete() 
    return render(request, 'index.html')

def chat(request):
    # if 'chat_history' in request.session:
    #     return redirect('/')
    
    chat_history = request.session.get('chat_history', [])
    
    if request.method == 'GET':
        return render(request, 'chat.html', {'chat_history': chat_history})
    
    elif request.method == 'POST':
        
        data = json.loads(request.body.decode('utf-8'))
        user_message = data.get('sentence')        
        bot_message = predict(user_message)
        
        chat_history.append(user_message)
        chat_history.append(bot_message)
        request.session['chat_history'] = chat_history

        return JsonResponse({'predicted_sentence': bot_message})
    
def question(request):
    request.session['chat_questionCheck'] = "YES"
    chat_history = request.session.get('chat_history', [])
    
    question = questionGet()

    chat_history.append(question)
    request.session['chat_history'] = chat_history
    
    return JsonResponse({'question': question})

def questionCheck(request):
    chat_questionCheck = request.session.get('chat_questionCheck')
    print('chat_questionCheck:' + chat_questionCheck)
   
    return JsonResponse({'chat_questionCheck': chat_questionCheck})

def result(request):
    # if 'chat_history' in request.session:
    #     return redirect('/')
    
    request.session.clear()
    QuestionDB.objects.all().delete() 
    mbti_type = 'INTJ'
    
    return render(request, 'result.html', {'mbti_type': mbti_type})

