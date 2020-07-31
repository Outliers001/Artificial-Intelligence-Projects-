FROM python:3.7-slim-stretch

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install gcc mono-mcs g++ git curl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

ADD requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

COPY service_account_iam.json service_account_iam.json
ENV GOOGLE_APPLICATION_CREDENTIALS="service_account_iam.json"


ADD train.py /app/train.py
ADD storage.py /app/storage.py
RUN chmod +x /app/train.py



ENTRYPOINT ["python"]
CMD ["/app/train.py"]
