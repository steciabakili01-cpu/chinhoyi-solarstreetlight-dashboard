FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python dependencies, including the GIS libraries.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the dashboard source and its data/assets folders.
COPY . .

# Render provides PORT at runtime; app.py reads it in server_config().
EXPOSE 10000

CMD ["python", "app.py"]
