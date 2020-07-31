FROM python:3.7-slim-stretch

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install gcc mono-mcs g++ git curl bash && \
    rm -rf /var/lib/apt/lists/*

ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

ADD servebreast.py /app/servebreast.py

COPY model/ app/model

# COPY . /app
WORKDIR /app

CMD ["python","servebreast.py"]


