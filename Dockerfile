FROM python:3.11.11-slim
# FROM python:3.10.12

ENV LANG=C.UTF-8 PYTHONIOENCODING=UTF-8 PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends python3-dev gcc rustc cargo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app/
COPY setup.py ./
RUN pip install .

COPY *.py ./
COPY README.md ./
COPY exec-*.sh ./

# RUN mkdir -p /opt/ssl/ && openssl req -x509 -newkey rsa:4096 -keyout /opt/ssl/tls.key -out /opt/ssl/tls.crt -days 9999 -nodes -subj "/CN=localhost"

CMD [ "/bin/bash", "-c", "./exec-prod.sh" ]
