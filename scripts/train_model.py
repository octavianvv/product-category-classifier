# train_model.py
# Scop: antreneaza un model ML care prezice categoria produsului din titlu
# Salveaza modelul + vectorizatorul TF-IDF in scripts/models/.

import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_and_clean(csv_path: str) -> tuple[pd.Series, pd.Series]:
    """
    Citim dataset-ul si il curatam minimal:
    - normalizam numele coloanelor (strip) ca sa evitam spatii in plus
    - pastram doar Product Title + Category Label
    - eliminam valori lipsa
    - normalizam textul
    """
    df = pd.read_csv(csv_path)

    # IMPORTANT: uneori CSV-ul are spatii in numele coloanelor
    df.columns = df.columns.str.strip()

    required = {"Product Title", "Category Label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"CSV-ul trebuie sa contina coloanele: {required}. "
            f"Am gasit: {list(df.columns)}"
        )

    df = df[["Product Title", "Category Label"]].copy()
    df.dropna(subset=["Product Title", "Category Label"], inplace=True)

    df["Product Title"] = df["Product Title"].astype(str).str.lower().str.strip()
    df["Category Label"] = df["Category Label"].astype(str).str.strip()

    X = df["Product Title"]
    y = df["Category Label"]
    return X, y


def main():
    # 1) Citim datele
    data_path = os.path.join("data", "products.csv")
    X, y = load_and_clean(data_path)

    # 2) Train/Test split (80/20) cu stratificare
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Vectorizare TF-IDF
    # Decizie: limitam vocabularul (max_features) ca sa fie rapid si stabil.
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # 4) Model: Logistic Regression (baseline solid pentru text)
    model = LogisticRegression(max_iter=2000, n_jobs=-1)
    model.fit(X_train_tfidf, y_train)

    # 5) Evaluare rapida (ca sa stim ca modelul "are sens")
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy (LogisticRegression + TF-IDF): {acc:.4f}\n")

    # zero_division=0 elimina warning-urile cand exista clase rare (support mic)
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 6) Salvam modelul + vectorizatorul in scripts/models/
    models_dir = os.path.join("scripts", "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "product_category_model.pkl")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(vec_path, "wb") as f:
        pickle.dump(tfidf, f)

    print("\nModelul si vectorizatorul au fost salvate cu succes:")
    print(f"- {model_path}")
    print(f"- {vec_path}")


if __name__ == "__main__":
    main()
