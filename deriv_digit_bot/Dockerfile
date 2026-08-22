FROM python:3.11-slim

WORKDIR /app

# Install deps first so Docker can cache this layer between builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the app
COPY . .

CMD ["python", "main.py"]
