# 라이브러리
pip install tensorflow_datasets
pip install django matplotlib tensorflow-datasets

#db 추가
python manage.py migrate
python manage.py makegrations

# 실행
cd chatbot
python manage.py runserver