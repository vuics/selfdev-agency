FROM python:3.12.9-slim-bookworm

ENV LANG=C.UTF-8 PYTHONIOENCODING=UTF-8 PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends python3-dev \
            curl ca-certificates gcc g++ make \
            npm wget libmagic1 chromium chromium-driver && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Rust using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set environment variables for Chrome/Chromium
ENV CHROME_BIN=/usr/bin/chromium \
    CHROMIUM_PATH=/usr/bin/chromium \
    BROWSER=/usr/bin/chromium \
    DISPLAY=:99

RUN npm i -g nodemon
RUN pip install --upgrade pip && \
    pip install nltk==3.9.1 && \
    python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger_eng

WORKDIR /opt/app/

# NOTE: The requirements.txt is to cache big number of dependencies.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# NOTE: The setup.py will contain additional dependencies.
# It helps for quick installation of new depenencies.
COPY setup.py ./
RUN pip install .

COPY src/*.py ./src/
COPY README.md ./
COPY input/* ./input/

# RUN pip install aiodns==3.2.0

# CMD [ "/bin/bash", "-c", "TBS" ]
