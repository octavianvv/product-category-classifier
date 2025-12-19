# predict_category.py
# Scop: incarca modelul + vectorizatorul TF-IDF si permite testare interactiva din consola.
# Utilizatorul introduce un titlu de produs, iar scriptul afiseaza categoria prezisa.

import os
import pickle


def load_artifacts():
    # Cautam fisierele salvate de train_model.py
    models_dir = os.path.join("scripts", "models")

    model_path = os.path.join(models_dir, "product_category_model.pkl")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Nu gasesc modelul: {model_path}. Ruleaza mai intai train_model.py."
        )
    if not os.path.exists(vec_path):
        raise FileNotFoundError(
            f"Nu gasesc vectorizatorul: {vec_path}. Ruleaza mai intai train_model.py."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vec_path, "rb") as f:
        tfidf = pickle.load(f)

    return model, tfidf


def main():
    model, tfidf = load_artifacts()

    print("Interactive mode: Product Category Prediction")
    print("Scrie un titlu de produs si apasa Enter.")
    print("Scrie 'exit' ca sa iesi.\n")

    while True:
        title = input("Titlu produs> ").strip()
        if title.lower() in {"exit", "quit", "q"}:
            print("Iesire. Succes!")
            break

        if not title:
            print("Te rog introdu un titlu (nu lasa gol).\n")
            continue

        # Normalizam la fel ca in train_model.py (lower + strip)
        title_clean = title.lower().strip()

        X = tfidf.transform([title_clean])
        pred = model.predict(X)[0]

        print(f"Categoria prezisa: {pred}\n")


if __name__ == "__main__":
    main()
