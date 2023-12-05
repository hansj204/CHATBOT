from .talkModel.talk_model import tokenizer, evaluate
import pandas as pd
from django.db import models
import pandas as pd
import numpy as np
import os
from googletrans import Translator
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re
import pickle
from sklearn.model_selection import train_test_split

nltk.download('punkt')
# 전역 변수 초기화
i_score = 0
e_score = 0
n_score = 0
s_score = 0
f_score = 0
t_score = 0
j_score = 0
p_score = 0

def predict(sentence):
    prediction = evaluate(sentence)
    predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence

def questionGet():

    data_path = 'mbti_chatbot/mbti_data/mbti_question.csv'
    df = pd.read_csv (data_path)
    question = df['Q'].sample(n = 1).reset_index(drop=True).iloc[0]

    print('question:' + question)  
    count = questionGetCheck(question)
    print(count)  

    questionDB= QuestionDB()
    if count == 0 :
        questionDB.question = question
        questionDB.save()
        return question
    else :   
        return questionGet()
    
def questionGetCheck(question):
    questionList = QuestionDB.objects.filter(question=question) 
    count = questionList.count()
    return count

# 텍스트 전처리 함수
def preprocess_text(text):
   
    # 번역
    translator = Translator()
    translated_text = translator.translate(text, src='ko').text
    print("번역된 텍스트:", translated_text)

    # 소문자 변환
    lower_text = translated_text.lower()
    print("소문자 변환된 텍스트:", lower_text)

    # 정규화
    normalized_text = re.sub('[^a-zA-Z0-9]', ' ', lower_text).strip()
    print("정규화된 텍스트:", normalized_text)

    # 토큰화
    tokens = nltk.word_tokenize(normalized_text)
    print("토큰화된 텍스트:", tokens)

    # 불용어 제거
    stop_words_list = stopwords.words('english')
    filtered_tokens = [word for word in tokens if word not in stop_words_list and word != 'think']
    print("불용어 제거된 텍스트:", filtered_tokens)

    # 원형화
    stemmer = SnowballStemmer(language='english')
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    print("원형화된 텍스트:", stemmed_tokens)

    return ' '.join(stemmed_tokens)

# 챗봇으로 받은 데이터 처리
def input_test(text):
    return preprocess_text(text)

def model_load():
    recreate_model = False
    print(os.getcwd())
    # filename = 'mbti_chatbot/resultModel/mbti_model.csv'
    filename = os.getcwd() + '\\mbti_chatbot\\resultModel\\mbti_model.sav'
    filename = filename.replace("\\", "/")
    filename = filename.replace("/", "\\")

    # 모델 로딩
    text_clf = pickle.load(open(filename, 'rb'))
    return text_clf
    
# 스코어링
def scoring(sentence):
    global i_score, e_score, n_score, s_score, f_score, t_score, j_score, p_score  # 전역 변수 사용 선언

    text_clf = model_load()
    a = sentence
    processed_text = input_test(a)

    # Series가 비어 있는지 확인
    if processed_text:
        predicted_mbti = text_clf.predict([processed_text])[0]
        print("예측된 MBTI 유형:", predicted_mbti)
        
         # 각 MBTI 차원별 점수 누적
        if predicted_mbti[0] == 'I':
            i_score += 1
        else:
            e_score += 1
        if predicted_mbti[1] == 'N':
            n_score += 1
        else:
            s_score += 1
        if predicted_mbti[2] == 'F':
            f_score += 1
        else:
            t_score += 1
        if predicted_mbti[3] == 'J':
            j_score += 1
        else:
            p_score += 1
            
        final_calculation()
    else:
        print("예측 불가: 입력된 텍스트가 유효하지 않음")
        
# 마지막에 호출되는 함수로 최종 MBTI 유형 및 퍼센트 계산
def final_calculation():
    global i_score, e_score, n_score, s_score, f_score, t_score, j_score, p_score

    # 각 요소별 퍼센트 계산
    total_ie = i_score + e_score
    total_ns = n_score + s_score
    total_ft = f_score + t_score
    total_jp = j_score + p_score

    ie_percentage = (i_score / total_ie) * 100 if total_ie else 0
    ns_percentage = (n_score / total_ns) * 100 if total_ns else 0
    ft_percentage = (f_score / total_ft) * 100 if total_ft else 0
    jp_percentage = (j_score / total_jp) * 100 if total_jp else 0

    # 결과 출력
    print(f"I/E Percentage: I - {ie_percentage}%, E - {100 - ie_percentage}%")
    print(f"N/S Percentage: N - {ns_percentage}%, S - {100 - ns_percentage}%")
    print(f"F/T Percentage: F - {ft_percentage}%, T - {100 - ft_percentage}%")
    print(f"J/P Percentage: J - {jp_percentage}%, P - {100 - jp_percentage}%")
    
    return {
        "I/E": f"I - {ie_percentage}%, E - {100 - ie_percentage}%",
        "N/S": f"N - {ns_percentage}%, S - {100 - ns_percentage}%",
        "F/T": f"F - {ft_percentage}%, T - {100 - ft_percentage}%",
        "J/P": f"J - {jp_percentage}%, P - {100 - jp_percentage}%"
    }

def analyze_mbti_scores(mbti_scores):
    result = {}
    threshold = 60  # 임계값 설정

    for dimension, score in mbti_scores.items():
        # 각 차원별 퍼센트 추출
        i_percentage, e_percentage = map(float, re.findall(r'\b\d+\.\d+|\d+\b', score))

        # 한쪽으로 치우친 경우
        if i_percentage > threshold or e_percentage > threshold:
            if i_percentage > e_percentage:
                dominant_letter = dimension[0]  # 예: 'I' in 'I/E'
                result[dimension] = (dominant_letter, "치우침")
            else:
                dominant_letter = dimension[-1]  # 예: 'E' in 'I/E'
                result[dimension] = (dominant_letter, "치우침")
        else:
            # 퍼센트가 비슷한 경우
            result[dimension] = ("균형", "비슷함")

    return result
    
def generate_mbti_explanation(analysis_result):
    explanations = []

    for dimension, (dominant_letter, status) in analysis_result.items():
        if status == "치우침":
            explanations.append(f"당신은 '{dimension}' 성향 중에서 '{dominant_letter}'에 치우쳐 있는 경향이 있습니다. {dimension_explanation[dominant_letter]}")
        else:
            explanations.append(f"당신의 '{dimension}' 성향은 비슷한 퍼센트를 가지고 있습니다. {dimension_explanation[dimension]}")

    return " ".join(explanations)

def determine_mbti_type(mbti_scores):
    mbti_type = ""
    threshold = 5  # 비슷한 경우를 결정하는 임계값

    for dimension, score in mbti_scores.items():
        i_percentage, e_percentage = map(float, re.findall(r'\b\d+\.\d+|\d+\b', score))

        # 비슷한 경우 처리
        if abs(i_percentage - e_percentage) <= threshold:
            mbti_type += "X"  # 'X'는 퍼센트가 비슷한 경우를 나타냄
        elif i_percentage > e_percentage:
            mbti_type += dimension[0]
        else:
            mbti_type += dimension[-1]

    return mbti_type
    
# 각 차원별 설명
dimension_explanation = {
    "I": "내향적(Introverted)이며, 혼자 있는 시간을 중요시하고 사색을 즐깁니다. 이러한 유형의 사람들은 내부 세계에 더 집중하며, 혼자 있는 것을 편안하게 느낍니다.",
    "E": "외향적(Extraverted)이며, 사람들과 어울리는 것을 선호하고 활동적입니다. E성향의 사람들은 외부 세계와의 상호작용을 통해 에너지를 얻습니다.",
    "N": "직관적(Intuitive)이며, 상상력이 풍부하고 새로운 가능성을 탐색하는 것을 좋아합니다. N유형의 사람들은 미래 지향적이며 추상적인 개념에 관심이 많습니다.",
    "S": "감각적(Sensing)이며, 현실적이고 구체적인 정보에 집중하는 경향이 있습니다. 이러한 유형의 사람들은 현재에 초점을 맞추고 감각적인 경험을 중시합니다.",
    "F": "감정적(Feeling)이며, 타인의 감정에 민감하고 조화를 중시합니다. 감정적인 가치와 인간 관계에 중점을 둡니다.",
    "T": "사고적(Thinking)이며, 논리적이고 객관적인 판단을 중요시합니다. T유형의 사람들은 사실과 원리를 기반으로 결정을 내립니다.",
    "J": "판단적(Judging)이며, 계획적이고 체계적인 접근을 선호합니다. 이러한 유형의 사람들은 조직된 방식으로 일을 처리하려고 합니다.",
    "P": "인식적(Perceiving)이며, 유연하고 개방적인 태도를 가지고 있습니다. 적응력이 있고 즉흥적인 결정을 선호합니다.",
    "I/E": "내향적이면서도 외향적인 성향을 적절히 가지고 있는 것을 의미합니다. 이는 사람이 상황에 따라 내향적이거나 외향적인 경향을 보일 수 있음을 나타냅니다.",
    "N/S": "상황에 따라 직관적인 접근(새로운 가능성 탐색, 추상적 사고)과 감각적인 접근(현실적인 정보에 기반한 구체적 사고) 사이에서 균형을 맞출 수 있습니다. 이러한 유형의 사람들은 큰 그림을 볼 수 있으면서도 세부 사항을 놓치지 않는 능력을 가지고 있습니다.",
    "F/T": "이러한 사람들은 감정적인 고려(타인의 감정과 가치에 민감함)와 사고적인 분석(논리적이고 객관적인 판단) 사이에서 균형을 이룰 수 있습니다. 이러한 유형의 사람들은 감정적인 요소와 논리적인 요소를 모두 고려하여 의사결정을 내릴 수 있습니다.",
    "J/P": "체계적으로 접근하는 면과 유연하고 적응적인 면, 양쪽 모두의 성향을 적절히 가지고 있습니다. 필요에 따라 체계적이고 조직적인 방식으로 일을 처리하거나 상황에 따라 융통성 있고 즉흥적으로 대응할 수 있습니다"
}
       
class QuestionDB(models.Model):
    question = models.CharField(max_length=500)