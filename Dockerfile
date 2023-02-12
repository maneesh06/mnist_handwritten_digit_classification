FROM jupyter/scipy-notebook

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN mkdir model
ENV MODEL_DIR=/home/jovyan/model
ENV MODEL_FILE=model.json
ENV WEIGHTS_FILE=model.h5

COPY train.py ./train.py
COPY inference.py ./inference.py

RUN python3 train.py