FROM tensorflow/tensorflow:2.19.0
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

EXPOSE 5000

CMD ["python", "main.py"]