FROM python:3.11-slim

# Install build tools needed for packages like xgboost
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and pre-downloaded wheels
COPY requirements.txt .
COPY wheels ./wheels

# ✅ Install from wheels first, then fallback to PyPI
RUN pip install --no-cache-dir --find-links=./wheels -r requirements.txt

# Copy the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
