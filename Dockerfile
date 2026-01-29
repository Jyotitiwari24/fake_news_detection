# Use small Python image
FROM python:3.10-slim

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stopwords inside container
RUN python -m nltk.downloader stopwords

# Copy project files
COPY . .

# Expose port (Render/Cloud uses 5000)
EXPOSE 5000

# Start app using production server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
