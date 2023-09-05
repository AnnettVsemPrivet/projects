FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3.8 python3.8-dev git python3-pip

RUN git clone https://github.com/AnnettVsemPrivet/projects.git

RUN pip install --no-cache-dir -r /projects/emotion_classification/requirements.txt

RUN   sed -i 's/asarray(t.exterior/asarray(t.exterior.coords/' /usr/local/lib/python3.8/dist-packages/descartes/patch.py

WORKDIR /projects/emotion_classification/py_modules

CMD ["python3", "async_tg_bot.py"]