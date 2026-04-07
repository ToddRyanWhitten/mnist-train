FROM tensorflow/tensorflow:2.21.0

RUN apt-get update && apt-get install -y git sudo && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir tensorflowjs

RUN useradd -m whittendata && \
    echo "whittendata ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER whittendata
WORKDIR /home/whittendata
COPY . .
