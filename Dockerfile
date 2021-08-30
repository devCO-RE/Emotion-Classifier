FROM python:3.6.9

WORKDIR /root/server-test


RUN apt update
RUN apt install -y default-jre
RUN apt install -y default-jdk

COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python3", "./server.py"]