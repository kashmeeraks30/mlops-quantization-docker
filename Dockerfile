FROM python:3.9
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY utils.py .
COPY predict.py .
COPY Models/ ./Models/

CMD ["python","predict.py"]