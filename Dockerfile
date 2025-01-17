FROM python:3.11.11-slim

ENV LANG=C.UTF-8 PYTHONIOENCODING=UTF-8 PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends python3-dev gcc rustc cargo npm && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN npm i -g nodemon

WORKDIR /opt/app/
COPY setup.py ./
RUN pip install .

# RUN mkdir -p ./src/
COPY src/*.py ./src/
COPY README.md ./

# RUN mkdir -p /opt/ssl/ && openssl req -x509 -newkey rsa:4096 -keyout /opt/ssl/tls.key -out /opt/ssl/tls.crt -days 9999 -nodes -subj "/CN=localhost"

# CMD [ "/bin/bash", "-c", "TBS" ]
