FROM tensorflow/tensorflow:1.13.1

RUN pip install numpy pandas
RUN mkdir -p app/.models/fine_tuned/model

ADD .models/uncased_L-24_H-1024_A-16/ /app/.models/uncased_L-24_H-1024_A-16/
ADD .models/fine_tuned/model/ /app/.models/fine_tuned/model/
ADD *.py /app/

WORKDIR /app

