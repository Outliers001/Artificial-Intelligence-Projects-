FROM python:3.7-slim-stretch

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install gcc mono-mcs g++ git curl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

ADD requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt



# # copy model
# #ADD imdbReviews.txt /opt/imdbReviews.txt
ADD dataextract.py /app/dataextract.py
RUN chmod +x /app/dataextract.py



ENTRYPOINT ["python"]
CMD ["/app/dataextract.py"]
