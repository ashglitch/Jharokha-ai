FROM python:3.11-slim
WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY . .
RUN useradd -m user && chown -R user /app
USER user
ENV JHAROKHA_PUBLIC=1 JHAROKHA_TILE=512
CMD gunicorn -b 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 900 app:app
