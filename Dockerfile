# 1. Use a lightweight Python base
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements and install (Optimized for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the application code
COPY . .

# 5. Expose the port
EXPOSE 8000

# 6. Run the server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]