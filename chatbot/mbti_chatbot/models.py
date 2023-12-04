from .talkModel.talk_model import tokenizer, evaluate
import pandas as pd
from django.db import models
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
class QuestionDB(models.Model):
    question = models.CharField(max_length=500)