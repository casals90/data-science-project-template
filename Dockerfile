FROM jupyter/base-notebook:latest

# expose default Jupyter port
EXPOSE 8888

COPY config/requirements.txt /app/config/requirements.txt
RUN pip install --upgrade pip setuptools
RUN pip install -r /app/config/requirements.txt

COPY config/settings.yaml /config/settings.yaml