# Alap image
FROM python:3.12-slim

# Munkakönyvtár beállítása
WORKDIR /app

# Rendszerfüggőségek telepítése (ha szükséges)
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Függőségek másolása
COPY requirements.txt .

# Pip frissítése
RUN pip install --upgrade pip

# Függőségek telepítése
RUN pip install --no-cache-dir -r requirements.txt

# Projekt másolása
COPY . .

# Alapértelmezett parancs
CMD ["python", "mlflow_experiment.py"]
