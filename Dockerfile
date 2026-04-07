FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir tensorflowjs

RUN useradd -m whittendata

USER whittendata
WORKDIR /home/whittendata
COPY . .
