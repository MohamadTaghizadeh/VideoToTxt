# Use Python 3.9 as base image
FROM python:3.9-slim

WORKDIR /approot/src
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ARG PIP_TIMEOUT=500
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
    
# Upgrade pip first
RUN python -m pip install --upgrade pip

# Install Python dependencies one by one to better handle errors







# Copy current directory contents into container
ADD ./requirements.txt /srv/
RUN python -m pip install -r /srv/requirements.txt --timeout $PIP_TIMEOUT



# Set the line endings to LF
RUN find . -type f -name "*.py" -exec sed -i 's/\r$//' {} +
