FROM python:3.10-slim

WORKDIR /app

# OpenCV video codecs + basic runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Create directories Flask needs at runtime
RUN mkdir -p /app/webapp/uploads /app/webapp/outputs /app/webapp/static

EXPOSE 5000

# Launch Flask dashboard
CMD ["python", "webapp/app.py"]