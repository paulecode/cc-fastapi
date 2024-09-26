FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip;

RUN pip install --no-cache-dir -r requirements.txt;

COPY . .

# EXPOSE 8000

# CMD ["uvicorn", "main:app"]

# CMD ["fastapi", "run"]

EXPOSE 10000

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","10000"]
