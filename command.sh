conda install django matplotlib tensorflow

django-admin startproject chatbot
cd chatbot
python manage.py migrate

python manage.py startapp mbti_chatbot

python manage.py runserver
