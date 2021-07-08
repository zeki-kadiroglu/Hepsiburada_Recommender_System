FROM ubuntu
WORKDIR /APP
COPY . .
RUN apt-get update -y
RUN apt-get install python3 -y
RUN apt-get install python3-pip -y
RUN pip3 install -r requirements.txt
EXPOSE 80
CMD [ "python3","main.py"]
