# product-category-classifier

Proiect de Machine Learning pentru clasificarea automată a produselor pe baza titlului produsului.

## Scop
Prezicerea categoriei unui produs (ex: Mobile Phones, TVs, Fridges etc.) folosind doar titlul acestuia.

## Structura proiectului

product-category-classifier/
├── data/
│ └── products.csv
├── notebooks/
│ └── product_category_analysis.ipynb
├── scripts/
│ ├── train_model.py
│ ├── predict_category.py
│ └── models/
│ ├── product_category_model.pkl
│ └── tfidf_vectorizer.pkl
├── README.md
└── .gitignore


## Metodologie
- Curatarea si normalizarea textului
- Vectorizare folosind TF-IDF
- Model de clasificare: Logistic Regression
- Impartire date: 80% train / 20% test (stratificat)

## Rezultate
- Acuratete: ~95%
- Performanta foarte buna pe categoriile principale
- Clasele cu putine exemple au rezultate mai slabe din cauza dezechilibrului datelor

## Rulare locala

Antrenarea modelului: python scripts/train_model.py

Predictie interactiva: python scripts/predict_category.py


## Notebook
Notebook-ul `product_category_analysis.ipynb` contine analiza exploratorie si evaluarea modelului.

## Autor
Octavian Ghita

