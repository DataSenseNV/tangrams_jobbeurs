FROM python:slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    libpq-dev \
    libssl-dev \
    odbc-postgresql \
    openssh-client \
    procps \
    unixodbc-dev \
    libgl1-mesa-glx \
    libgtk2.0-dev

RUN rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["python"]