FROM python:3.14.7-slim 

WORKDIR /app

# Refresh package metadata, install build tools without recommended extras, then remove cached package lists.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the backend application package into the image.
COPY ./app ./app 

EXPOSE 8000

# Start the backend by running app.main as a Python module.
CMD ["python", "-m", "app.main"]