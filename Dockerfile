FROM python:3.11.11-slim

ENV LANG=C.UTF-8 PYTHONIOENCODING=UTF-8 PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends python3-dev gcc rustc cargo npm wget libmagic1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Trying to fix the issue with the missing browser
#
#RUN apt-get update --yes && \
#    apt-get upgrade --yes && \
#    apt-get install --yes --no-install-recommends python3-dev gcc rustc cargo npm wget
#RUN apt-get install --yes --no-install-recommends chromium
##RUN apt-get install --yes --no-install-recommends chromium-browser
## The Chrome repository doesn’t provide arm64 packages:
##
## RUN apt-get install --yes --no-install-recommends gnupg
## RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \ 
##     && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list
## RUN apt-get update && apt-get -y install google-chrome-stable
#RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN npm i -g nodemon
RUN pip install --upgrade pip && \
    pip install nltk==3.9.1 && \
    python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger_eng

WORKDIR /opt/app/
COPY setup.py ./
RUN pip install .

COPY src/*.py ./src/
COPY README.md ./
COPY input/* ./input/


# RUN mkdir -p /opt/ssl/ && openssl req -x509 -newkey rsa:4096 -keyout /opt/ssl/tls.key -out /opt/ssl/tls.crt -days 9999 -nodes -subj "/CN=localhost"

# CMD [ "/bin/bash", "-c", "TBS" ]
