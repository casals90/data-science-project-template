# Base image with JupyterLab
FROM jupyter/base-notebook:lab-4.0.2

# Expose the default Jupyter port
EXPOSE 8888

# Set up Poetry environment variables
ENV POETRY_VERSION=1.5.1
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install Poetry
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && chmod +x $POETRY_HOME/bin/poetry

# Ensure Poetry is available and configure it
RUN $POETRY_HOME/bin/poetry config virtualenvs.create false

# Set the working directory
WORKDIR /app

# Copy Poetry configuration files
# COPY config/pyproject.toml config/poetry.lock* /app/config
COPY ./config/pyproject.toml ./config/poetry.lock* /app/

RUN echo ls -l /

# Install project dependencies using Poetry
RUN $POETRY_HOME/bin/poetry install --no-root

# Copy additional necessary files
COPY config/settings.yaml /config/settings.yaml
COPY . /app

# Switch back to the default user
USER $NB_UID
