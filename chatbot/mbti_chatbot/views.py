from django.shortcuts import render, redirect
from .models import predict, scoring, questionGet, QuestionDB, final_calculation, analyze_mbti_scores, generate_mbti_explanation, determine_mbti_type, simplify_mbti_scores
from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt

def index(request):
    request.session['chat_history'] = []
    request.session['chat_init_history'] = []
    request.session['chat_questionCheck'] = "NO"
    QuestionDB.objects.all().delete() 
    return render(request, 'index.html')

@csrf_exempt
def chat(request):
    chat_history = request.session.get('chat_history', [])
    chat_init_history = request.session.get('chat_init_history', [])
    if request.method == 'GET':
        return render(request, 'chat.html', {'chat_history': chat_history , 'chat_init_history': chat_init_history})
    
    elif request.method == 'POST':
        
        data = json.loads(request.body.decode('utf-8'))
        user_message = data.get('sentence')        
        bot_message = predict(chat_history[-1], user_message)
        
        chat_history.append(user_message)
        chat_history.append(bot_message)
        request.session['chat_history'] = chat_history
        
        # 스코어링
        scoring(user_message)

        return JsonResponse({'predicted_sentence': bot_message})
    
def question(request):
    request.session['chat_questionCheck'] = "YES"
    
    chat_history = request.session.get('chat_history', [])
    
    question = questionGet()

    chat_history.append(question)
    request.session['chat_history'] = chat_history
    
    return JsonResponse({'question': question})
    
def initMsg(request):
    request.session['chat_questionCheck'] = "YES"
    chat_init_history = request.session.get('chat_init_history', [])
   
    chat_init_history.append("안녕하세요!<br>저는 당신의 대답을 바탕으로 MBTI 성격 유형을 분석합니다.")
    chat_init_history.append("간단한 대화를 통해 자신에 대해 더 많이 알아가 보세요.")
    chat_init_history.append("제가 드리는 질문에 답해주신다면 보내주시는 답변을 통해 당신의 MBTI 유형을 파악하고 결과를 이미지로 저장할 수도 있어요.")
    
    request.session['chat_init_history'] = chat_init_history
    
    return JsonResponse({'initMsg': chat_init_history})
    
def questionCheck(request):
    chat_questionCheck = request.session.get('chat_questionCheck')
    return JsonResponse({'chat_questionCheck': chat_questionCheck})

def result(request):
    request.session.clear()
    QuestionDB.objects.all().delete() 
    
    # MBTI 성향 계산
    mbti_scores = final_calculation()
    simple_mbti_scores = simplify_mbti_scores(mbti_scores)
    
    # MBTI 유형 결정
    mbti_type = determine_mbti_type(mbti_scores)

    # 성향 분석
    analysis_result = analyze_mbti_scores(mbti_scores)

    # 결과 설명 생성
    explanation = generate_mbti_explanation(analysis_result)
    
    print("도출된 mbti:", mbti_type)
    print("퍼센트:", simple_mbti_scores)
    print("퍼센트에 따른 설명:", explanation)

    # 결과 페이지에 정보 전달
    return render(request, 'result.html', {'mbti_type': mbti_type, 'simplify_mbti_scores': simple_mbti_scores, 'mbti_explanation': explanation})

