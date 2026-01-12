# 1. Use a specific, stable Debian version (Bullseye) to avoid apt errors
FROM python:3.9-slim-bullseye

# 2. Prevent "interactive" prompts from stopping the build
ENV DEBIAN_FRONTEND=noninteractive

# 3. Update & Install with "--fix-missing" to prevent Error 100
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Setup App
WORKDIR /app

# 5. Install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Code & Start
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
