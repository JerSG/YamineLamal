# Reproduce the submission in a pinned environment:
#   docker build -t yaminelamal .
#   docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/submission:/app/submission" yaminelamal
# The match data is mounted read-only at run time; it is never copied into the image.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "run.py", "--no-open"]
