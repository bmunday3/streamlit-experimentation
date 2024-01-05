FROM python:3.9-slim-bullseye

RUN apt-get update \
    && apt-get install libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# set directory for app
WORKDIR /opt/app

# define argument for port
ENV PORT=8000

# copy requirements file
COPY ./requirements.txt .
# pip install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy additional flask app files
COPY ./imgs/ imgs/
COPY ./__init__.py __init__.py
COPY ./streamer.py streamer.py
COPY ./app.py app.py

# entrypoint
CMD streamlit run app.py --server.port=$PORT --server.address="0.0.0.0"