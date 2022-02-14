FROM python:3.7

WORKDIR /usr/src/app

RUN apt-get update 
RUN apt-get install libgl1 -y

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .

CMD sh -c "python 0_fetch_from_database.py && python 1_splitting_into_patches.py"
