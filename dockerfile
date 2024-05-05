FROM python:3.10

WORKDIR /recognition_entity

COPY . /recognition_entity

RUN pip install -r /recognition_entity/requirements.txt

COPY ./app_recognition /recognition_entity/app

EXPOSE 8000

CMD ["uvicorn", "app_recognition.fast_api:app", "--host", "0.0.0.0", "--port", "8000"]