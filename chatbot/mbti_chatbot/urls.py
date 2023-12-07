from django.urls import path

from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('chat', views.chat, name='chat'),
    path('result', views.result, name='result'),
    path('question', views.question, name='question'),
    path('questionCheck', views.questionCheck, name='questionCheck'),
    path('initMsg', views.initMsg, name='initMsg'),
]