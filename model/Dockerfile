FROM python:3.7

WORKDIR /usr/src/app

RUN apt-get update 
RUN apt-get install libgl1 -y

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD sh -c "python train.py --config config.json"
