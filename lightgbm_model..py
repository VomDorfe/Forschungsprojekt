# === IMPORTE ===
import pandas as pd
import numpy as np
import re
import nltk
import time
import shap
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier, early_stopping
from datetime import datetime

# === STARTZEIT MESSUNG ===
start_time = time.time()

# === NLTK RESSOURCEN LADEN ===
nltk.download('stopwords')
nltk.download('wordnet')

# === DATEN LADEN UND BEREINIGEN ===
csv_path = "/Users/lassewesterbuhr/Documents/Uni/Bachelor/8.Semester/Forschungsprojekt/Import/IMDB Dataset.csv"
df = pd.read_csv(csv_path)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>|[^a-zA-Z\s]', '', text).lower()
    tokens = [lemmatizer.lemmatize(word) for word in text.split()
              if (word not in stop_words) and (len(word) > 2)]
    return ' '.join(tokens[:400])

df['review_clean'] = df['review'].apply(clean_text)

# === DATENAUFTEILUNG ===
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['review_clean'],
    df['sentiment'],
    test_size=0.2,
    stratify=df['sentiment'],
    random_state=42
)

# === TF-IDF VEKTORISIERUNG ===
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=5,
    max_df=0.9
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Für SHAP als DataFrame mit Feature-Namen
feature_names = vectorizer.get_feature_names_out()
X_train_df = pd.DataFrame(X_train.toarray(), columns=feature_names)
X_test_df = pd.DataFrame(X_test.toarray(), columns=feature_names)

# === HYPERPARAMETER-TUNING ===
param_dist = {
    'learning_rate': [0.1],
    'num_leaves': [63],
    'max_depth': [6],
    'n_estimators': [1000],
    'min_data_in_leaf': [10],
    'bagging_fraction': [0.8],
    'feature_fraction': [0.8],
    'min_split_gain': [0.0],
}

lgb_model = LGBMClassifier(
    objective='binary',
    random_state=42,
    verbosity=-1,
    n_jobs=1,
    n_estimators=2000,
    callbacks=[early_stopping(stopping_rounds=20)]
)

random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_dist,
    n_iter=100,
    scoring='f1',
    cv=2,
    verbose=2,
    random_state=42,
    n_jobs=1
)

random_search.fit(X_train, y_train)

# === SPEICHERE ERGEBNISSE ===
def save_search_results_txt(search_object, filename):
    with open(filename, "w") as f:
        f.write("Beste Parameter:\n")
        for param, value in search_object.best_params_.items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nBestes Ergebnis (Score): {search_object.best_score_:.4f}\n")

        f.write("\nAlle getesteten Parameterkombinationen und Scores:\n")
        for i, res in enumerate(search_object.cv_results_['params']):
            score = search_object.cv_results_['mean_test_score'][i]
            f.write(f"{i + 1}. {res} | Score: {score:.4f}\n")
    print(f"Ergebnisse gespeichert in: {filename}")

save_search_results_txt(random_search, "lgbm_random_search_results8.txt")

# === EVALUATION ===
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === SHAP ANALYSE ===
# === DataFrame aus Sparse-Matrix erstellen (falls noch nicht geschehen) ===
X_train_df = pd.DataFrame.sparse.from_spmatrix(X_train, columns=vectorizer.get_feature_names_out())
X_test_df = pd.DataFrame.sparse.from_spmatrix(X_test, columns=vectorizer.get_feature_names_out())

# === SHAP Explainer und Werte berechnen ===
import datetime
explainer = shap.Explainer(best_model, X_train_df)
shap_values = explainer(X_test_df)

# === Funktion: Extremfall auswählen ===
def get_example_by_confidence(shap_values, explainer, type='high'):
    fx = shap_values.values.sum(axis=1) + explainer.expected_value
    if type == 'high':
        idx = np.argmax(fx)
    elif type == 'low':
        idx = np.argmin(fx)
    elif type == 'mid':
        idx = np.argmin(np.abs(fx - 0.5))
    else:
        raise ValueError("Ungültiger Typ: 'high', 'low' oder 'mid'")
    return idx, fx[idx]

# === Funktion: Plot & Text speichern ===
def save_decision_plot(idx, score, label, shap_values, explainer, X_test_df, X_test_text, df, best_model):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    shap_val = shap_values[idx]
    cleaned_text = X_test_text.iloc[idx]
    original_index = X_test_text.index[idx]
    original_text = df.loc[original_index, "review"]
    prediction = best_model.predict([X_test_df.iloc[idx]])[0]
    prediction_label = "positiv" if prediction == 1 else "negativ"

    # === SHAP Decision Plot erzeugen ===
    shap.plots.waterfall(shap_val, max_display=20, show=False)
    filename_plot = f"shap_decisionplot_{label}_{timestamp}.png"
    plt.savefig(filename_plot, bbox_inches="tight")
    plt.close()

    # === SHAP Textbericht ===
    filename_txt = f"shap_review_{label}_{timestamp}.txt"
    with open(filename_txt, "w", encoding="utf-8") as f:
        f.write(f"SHAP Decision Plot: {label.capitalize()} Confidence\n")
        f.write(f"Index im Testset: {idx}\n\n")

        f.write("=== Original Review (unbereinigt) ===\n")
        f.write(original_text.strip() + "\n\n")

        f.write("=== Bereinigter Text ===\n")
        f.write(cleaned_text.strip() + "\n\n")

        f.write("=== Modellvorhersage ===\n")
        f.write(f"f(x) = {score:.4f}\n")
        f.write("→ Klassifikation: " + prediction_label + "\n\n")

        f.write("=== Top SHAP-Wörter ===\n")
        for name, val in zip(shap_val.feature_names[:20], shap_val.values[:20]):
            f.write(f"{name}: {val:+.3f}\n")

    print(f"→ Gespeichert: {filename_plot} & {filename_txt}")

# === Anwendung auf drei Fälle ===
for label in ["high", "low", "mid"]:
    idx, score = get_example_by_confidence(shap_values, explainer, label)
    save_decision_plot(idx, score, label, shap_values, explainer, X_test_df, X_test_text, df, best_model)


# === SHAP Summary Plot ===
# === SHAP-Explainer erstellen (TreeExplainer empfohlen für LightGBM) ===
#explainer = shap.Explainer(best_model, X_train_df)
# === SHAP-Werte für gesamten Testdatensatz berechnen ===
#shap_values = explainer(X_test_df)
# === Zeitstempel für Dateinamen erzeugen ===
#timestamp = datetime.now().strftime("%Y%m%d_%H%M")
# === Summary Plot erzeugen und speichern ===
#shap.summary_plot(shap_values, max_display=20, show=False)
#plt.tight_layout()
#plt.savefig(f"shap_summary_lightgbm_fullset_{timestamp}.png", bbox_inches="tight", dpi=300)

# === Bar Plot
# Erzeuge Zeitstempel im Format YYYYMMDD_HHMM
#timestamp = datetime.now().strftime("%Y%m%d_%H%M")
# SHAP-Werte berechnen (falls noch nicht erfolgt)
#explainer = shap.Explainer(best_model, X_train_df)
#shap_values = explainer(X_test_df)
# Barplot speichern mit Zeitstempel
#plt.title("Globale Feature-Wichtigkeit (SHAP Bar Plot)")
#shap.plots.bar(shap_values, max_display=20, show=False)
#plt.tight_layout()
#plt.savefig(f"shap_barplot_lightgbm_{timestamp}.png", dpi=300, bbox_inches="tight")


# === GESAMTLAUFZEIT ===
print(f"\nGesamtlaufzeit: {round(time.time() - start_time, 2)} Sekunden")