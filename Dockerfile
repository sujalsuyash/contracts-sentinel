# 1. Use lightweight Python
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Upgrade pip to the latest version (uses less memory)
RUN pip install --upgrade pip

# 4. Copy requirements
COPY requirements.txt .

# --- THE MEMORY FIX ---
# We install the massive PyTorch library ALONE first.
# This gives it 100% of the available RAM.
RUN pip install --no-cache-dir torch==2.2.0 --extra-index-url https://download.pytorch.org/whl/cpu

# 5. Install the rest of the libraries
# (Pip will skip torch because it's already there)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your application code
COPY . .

# 7. Expose port
EXPOSE 8000

# 8. Start the server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]