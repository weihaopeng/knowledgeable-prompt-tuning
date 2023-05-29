FROM python:3.11.3-ubuntu20.04

LABEL maintainer="Moyu <Weihp16@gmail.com>"

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip setuptools wheel google-api-python-client python-dotenv -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir langchain==0.0.179 -i https://pypi.org/simple && \
    mkdir -p /code && \
    mkdir -p /data/knowledgeable-prompt-tuning/logs

WORKDIR /code

ENV \
PYTHONPATH=$PYTHONPATH:/code \
PORT=80 \
log_file=/data/knowledgeable-prompt-tuning/logs/qa.log \
LOG_LEVEL=INFO

COPY . /code/

CMD ["python", "main.py"]
