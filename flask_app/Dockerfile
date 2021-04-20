
# Getting base OS image
FROM python:3.6.1-alpine

#MAINTAINER Omkar Reddy <gojala.o@northeastern.edu>

#RUN apt-get update -y && \
#    apt-get install -y python3.7 python3-pip python-dev

# We copy just the requirements.txt first to leverage Docker cache
#COPY ./requirements.txt /app/requirements.txt

RUN pip install cython && pip3 install numpy

WORKDIR /image_caption_flask_app

ADD . /image_caption_flask_app

RUN pip install -r requirements.txt

CMD ["python", "app.py" ]
