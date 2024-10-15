FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

WORKDIR /app

RUN apt update && apt install -y -V \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libffi-dev \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    libpq-dev \
    libssl-dev \
    lsb-release \
    odbc-postgresql \
    openssh-client \
    procps \
    python3 \
    python3-pip \
    unixodbc-dev \
    wget

RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb

RUN apt update && apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb

RUN apt update && apt install -y -V \
    libarrow-dataset-dev \
    libarrow-dev \
    libarrow-flight-dev \
    libprotobuf-dev \
    protobuf-compiler

RUN rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN cat requirements.txt

RUN python3 -m pip install -r requirements.txt

CMD ["python3"]