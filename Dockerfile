FROM nvidia/cuda:10.2-base-ubuntu18.04

WORKDIR /app

RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get install -y python3.7 python3-distutils python3-pip python3-apt
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip3 install -U pip

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt

COPY . .

CMD ["python3", "main.py"]
