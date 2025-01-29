FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["streamlit", "run", "web_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]