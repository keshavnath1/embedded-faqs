FROM python:3.8
WORKDIR /src

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . /src
ENV FLASK_APP=app
CMD ["python","app.py"]
