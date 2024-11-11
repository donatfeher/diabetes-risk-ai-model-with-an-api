# FastAPI alapú gépi tanulás modell dokumentációja

[English version (Angol verzió)](README_en.md)

Az alábbi dokumentáció alapján mások is könnyen futtathatják a FastAPI alapú gépi tanulás modellt. Íme a lépések, hogyan futtassák a kódot a saját környezetükben.

## Projekt Struktúra

Az alábbiakban a projekt fontosabb fájljainak és mappáinak áttekintése:

```plaintext
.
├── README.md                # Projekt dokumentáció (angolul)
├── main.py                  # FastAPI szerver fő belépési pontja
├── requirements.txt         # Függőségek listája
├── data                     # Adatokat tartalmazó mappa
│   ├── diabetes.csv         # Eredeti adatállomány
│   └── processed_data.csv   # Előfeldolgozott adatállomány
├── models                   # Modellek mappája
│   ├── final_model.pkl      # Legjobb modell mentett verziója
│   └── model_pipeline.py    # Modell betanításáért és mentéséért felelős szkript
├── src                      # Segédfájlokat tartalmazó mappa
│   └── data_pipeline.py     # Adat előfeldolgozási funkciók
└── docs                     # Dokumentációs fájlok és képek
    ├── all_cicd_pipeline_success.png
    ├── final_result_prediction_success.png
    └── mlops_kurzus_tematikaja.txt
```

## 1. Klónozd vagy másold a kódot a helyi gépre

Győződj meg róla, hogy a **main.py** és a szükséges modellfájlok (például `final_model.pkl`) elérhetők a könyvtárban, ahol dolgozol.

## 2. Telepítsd a szükséges csomagokat

Lépj be abba a könyvtárba, ahol a `requirements.txt` található, majd telepítsd a szükséges Python-csomagokat a következő parancs segítségével:

```bash
pip install -r requirements.txt
```

Alternatív megoldásként az alábbi parancs futtatásával egyenként telepítheted a csomagokat:

```bash
pip install fastapi==0.112.2 uvicorn==0.30.0 scikit-learn==1.5.1 pandas==2.2.2 numpy==1.25.0 xgboost==2.1.1
```

## 3. Futtasd a FastAPI szervert

Az alábbi parancs segítségével indíthatod el a FastAPI szervert. Figyelj, hogy a `main` a fájl neve, ahol a FastAPI alkalmazásod található:

```bash
uvicorn main:app --reload
```

Ezzel elindítod a szervert, amely figyel a `http://127.0.0.1:8000` címen.

## 4. API dokumentáció

A FastAPI automatikusan generál egy interaktív dokumentációs felületet. A böngésződben érheted el az alábbi URL-en:

```plaintext
http://127.0.0.1:8000/docs
```

Itt kipróbálhatod az API végpontjait és megnézheted a várt bemeneti és kimeneti formátumokat.

## 5. Predikció küldése

A predikcióhoz használj egy `POST` kérést a következő URL-re:

```plaintext
http://127.0.0.1:8000/predict
```

Példa kérést küldhetsz a következő JSON-nel:

```json
{
  "Pregnancies": 2,
  "Glucose": 120.0,
  "BloodPressure": 70.0,
  "SkinThickness": 20.0,
  "Insulin": 85.0,
  "BMI": 25.0,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 30
}
```

## 6. Szerver válasza

A szerver a predikciót JSON formátumban adja vissza, például így:

```json
{
  "prediction": 1
}
```

## Gyakori hibák és megoldások

1. **`ModuleNotFoundError: No module named 'fastapi'`**: Ellenőrizd, hogy a szükséges csomagokat telepítetted a Python környezetedbe a `pip install -r requirements.txt` parancs segítségével.

2. **`ValueError: X has 8 features, but RandomForestClassifier is expecting 9 features`**: A modelled több bemeneti adatot vár. Ellenőrizd a modell működését és a bemeneti adatokat.

3. **Port foglaltság**: Ha a `8000` port foglalt, próbálj egy másik portot használni:

   ```bash
   uvicorn main:app --reload --port 8080
   ```