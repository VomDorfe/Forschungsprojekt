import pandas as pd
import re
import nltk
import numpy as np
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier


start_time = time.time()
nltk.download('stopwords')
nltk.download('wordnet')

# === Daten laden & bereinigen ===
csv_path = "/content/drive/MyDrive/Colab Notebooks/IMDB Dataset.csv"
df = pd.read_csv(csv_path)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>|[^a-zA-Z\s]', '', text).lower()
    tokens = [lemmatizer.lemmatize(word) for word in text.split()
             if (word not in stop_words) and (len(word) > 2)]
    return ' '.join(tokens[:400])  # Kürzere Texte

df['review_clean'] = df['review'].apply(clean_text)

# === Datenaufteilung ===
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['review_clean'],
    df['sentiment'],
    test_size=0.2,
    stratify=df['sentiment'],
    random_state=42
)

# === Kritische Änderung 3: Sparsame Feature-Extraktion ===
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=5,
    max_df=0.9
)

X_train = vectorizer.fit_transform(X_train_text)  # Bleibt sparse
X_test = vectorizer.transform(X_test_text)

# === Kritische Änderung 4: GPU-Parameter für LightGBM ===
param_dist = {
    'learning_rate': [0.02, 0.05, 0.08],
    'num_leaves': [127, 255, 511],
    'max_depth': [-1, 10, 15],
    'min_child_samples': [30, 50],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'reg_alpha': [0, 0.1, 0.2],
    'reg_lambda': [0, 0.1, 0.2],
    'n_estimators': [400, 600],
    'scale_pos_weight': [1, 2]  # Für Klassenbalance
}


lgb_model = LGBMClassifier(
    objective='binary',
    random_state=42,
    verbosity=-1,
    n_jobs=1  # Wichtig für Stabilität!
)

# === Kritische Änderung 5: Reduzierte Parallelität ===
random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_dist,
    n_iter=15,
    scoring='f1',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=1
)

random_search.fit(X_train, y_train)

# === Auswertung ===
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print(f"\nGesamtlaufzeit: {time.time() - start_time:.2f} Sekunden")
