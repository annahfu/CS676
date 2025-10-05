FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
COPY credrefactor.py /app/credrefactor.py
COPY server.py /app/server.py
COPY requirements_prod.txt /app/requirements_prod.txt
COPY models /app/models
RUN pip install --no-cache-dir -r requirements_prod.txt
EXPOSE 7860
CMD ['uvicorn','server:app','--host','0.0.0.0','--port','7860']
