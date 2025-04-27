# Use Python 3.9 as base image
FROM python:3.9-slim

ENV LANG=C.UTF-8
ARG PIP_TIMEOUT=500



# Install system dependencies








# Upgrade pip first
RUN python -m pip install --upgrade pip

# Copy requirements first for better caching
ADD ./requirements.txt /srv/
# Install Python dependencies
RUN python -m pip install -r /srv/requirements.txt --timeout $PIP_TIMEOUT


# Expose the port the app runs on
WORKDIR /approot/src
