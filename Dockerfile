FROM python:3.10-slim

WORKDIR /app

# Salin requirements dan install
COPY Membangun_model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode
COPY . .

# Perintah default saat docker dijalankan
CMD ["python", "Membangun_model/modelling.py"]
