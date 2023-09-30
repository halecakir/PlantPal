FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

# working directory
WORKDIR /app

# Install python3.9
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.9 python3.9-dev python3.9-distutils

RUN python3.9 -m pip install pip --upgrade 

# Install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN python3.9 -m pip install -r /tmp/requirements.txt

# Copy source code
COPY . /app


