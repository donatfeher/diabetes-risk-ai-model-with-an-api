Az alábbi dokumentáció alapján mások is könnyen futtathatják a FastAPI alapú gépi tanulás modellt. Íme a lépések, hogyan futtassák a kódot a saját környezetükben.

### 1. Klónozd vagy másold a kódot a helyi gépre

Győződj meg róla, hogy a **main.py** és a szükséges modellfájlok (például `final_model.pkl`) elérhetők a könyvtárban, ahol dolgozol.

### 2. Telepítsd a szükséges csomagokat

Lépj be a terminálba, és lépj be abba a könyvtárba, ahol a FastAPI fájlod található. Telepítened kell néhány Python-csomagot:

```bash
pip install fastapi uvicorn scikit-learn numpy xgboost
```

### 3. Futtasd a FastAPI szervert

Az alábbi parancs segítségével indíthatod el a FastAPI szervert. Figyelj, hogy a `main` a fájl neve, ahol a FastAPI alkalmazásod található:

```bash
uvicorn main:app --reload
```

Ezzel elindítod a szervert, amely figyel a `http://127.0.0.1:8000` címen.

### 4. API dokumentáció

A FastAPI automatikusan generál egy interaktív dokumentációs felületet. A böngésződben érheted el az alábbi URL-en:

```bash
http://127.0.0.1:8000/docs
```

Itt kipróbálhatod az API végpontjait.

### 5. Predikció küldése

A predikcióhoz használj egy `POST` kérést a következő URL-re:

```bash
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

### 6. Szerver válasza

A szerver a predikciót JSON formátumban adja vissza, például így:

```json
{
  "prediction": 1
}
```

### Gyakori hibák és megoldások

1. **`ModuleNotFoundError: No module named 'fastapi'`**: Ellenőrizd, hogy a szükséges csomagokat telepítetted a Python környezetedbe (`pip install fastapi uvicorn`).
   
2. **`ValueError: X has 8 features, but RandomForestClassifier is expecting 9 features`**: A modelled több bemeneti adatot vár. Ellenőrizd a modell működését és a bemeneti adatokat.

3. **Port foglaltság**: Ha a `8000` port foglalt, próbálj egy másik portot használni:
   
   ```bash
   uvicorn main:app --reload --port 8080
   ```

### Összegzés

Ezekkel a lépésekkel mások is könnyen futtathatják a kódodat, és használhatják a gépi tanulási modellt FastAPI-n keresztül. Ha további kérdés merülne fel, kérlek, jelezd!

