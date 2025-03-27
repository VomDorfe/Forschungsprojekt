import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split

# === 1. NLTK Setup ===
nltk.data.path.append("/Users/lassewesterbuhr/nltk_data")  # Stelle sicher, dass Punkt, Stopwords, WordNet geladen sind
nltk.download('stopwords', download_dir="/Users/lassewesterbuhr/nltk_data")
nltk.download('wordnet', download_dir="/Users/lassewesterbuhr/nltk_data")

# === 2. CSV-Pfad setzen ===
csv_path = "/Users/lassewesterbuhr/Documents/Uni/Bachelor/8.Semester/Forschungsprojekt/Import/IMDB Dataset.csv"  # anpassen falls nötig

# === 3. Daten laden ===
df = pd.read_csv(csv_path)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# === 4. Textbereinigung + Lemmatisierung ===
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

def clean_text(text, remove_stopwords=True):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# === 5. Zwei Varianten: mit und ohne Stopwords ===
df['review_clean'] = df['review'].apply(lambda x: clean_text(x, remove_stopwords=True))
df['review_nostop'] = df['review'].apply(lambda x: clean_text(x, remove_stopwords=False))

# === 6. Trainings-/Test-Split (80/20) ===
X_with_stop = df['review_clean']
X_without_stop = df['review_nostop']
y = df['sentiment']

X_train_with, X_test_with, y_train, y_test = train_test_split(X_with_stop, y, test_size=0.2, random_state=42, stratify=y)
X_train_no, X_test_no, _, _ = train_test_split(X_without_stop, y, test_size=0.2, random_state=42, stratify=y)

# === 7. Kontrollausgabe ===
print("Sample (Stopwords entfernt):", X_train_with.iloc[0][:300])
print("Sample (Stopwords behalten):", X_train_no.iloc[0][:300])
print("Train/Test sizes:", len(X_train_with), len(X_test_with))

from sklearn.feature_extraction.text import TfidfVectorizer

# === 1. TF-IDF-Vektorisierung (auf "X_train_with", also bereinigt & ohne Stopwords)
vectorizer = TfidfVectorizer(max_features=10_000)

# Fit auf Trainingsdaten → wichtig: Nur auf Training fitten, nicht auf Test!
X_train_tfidf = vectorizer.fit_transform(X_train_with)
X_test_tfidf = vectorizer.transform(X_test_with)

# === 2. Ausgabe: Form der Matrizen
print("Trainingsdaten (TF-IDF) Shape:", X_train_tfidf.shape)
print("Testdaten (TF-IDF) Shape:", X_test_tfidf.shape)

# === 3. Beispiel: Vektor eines einzelnen Texts
print("Erster Trainings-Vektor (gekürzt):", X_train_tfidf[0].toarray()[0][:20])

import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# === 1. LightGBM Dataset erstellen (optional, aber effizient)
train_data = lgb.Dataset(X_train_tfidf, label=y_train)
test_data = lgb.Dataset(X_test_tfidf, label=y_test, reference=train_data)

# === 2. Basis-Konfiguration
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1
}

# === 3. Modell trainieren (10 Runden reichen fürs Base-Modell)
model = lgb.train(
    params,
    train_data,
    num_boost_round=10,
    valid_sets=[test_data],
    valid_names=['test'],
)

# === 4. Vorhersage & Schwellenwert setzen
y_pred_probs = model.predict(X_test_tfidf)
y_pred = (y_pred_probs >= 0.5).astype(int)

# === 5. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import shap
import matplotlib.pyplot as plt

# Explainer erstellen
explainer = shap.Explainer(model, X_train_tfidf.toarray(), feature_names=vectorizer.get_feature_names_out())

# SHAP-Werte berechnen (für 100 Testbeispiele)
shap_values = explainer(X_test_tfidf[:100].toarray())

# Plot erzeugen
shap.summary_plot(shap_values, max_display=20, show=False)
plt.savefig("shap_summary_lightgbm.png", bbox_inches='tight')

# ----------hier beginnt LSTM-----------

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === 1. Parameter setzen ===
max_vocab_size = 10000  # maximale Anzahl Wörter, die behalten werden
max_sequence_length = 200  # alle Texte auf 200 Tokens kürzen/auffüllen

# === 2. Tokenizer initialisieren ===
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_with)

# === 3. Texte in Sequenzen umwandeln ===
X_train_seq = tokenizer.texts_to_sequences(X_train_with)
X_test_seq = tokenizer.texts_to_sequences(X_test_with)

# === 4. Padding auf gleiche Länge ===
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post', truncating='post')

# === 5. Zielwerte als Numpy-Array (für Keras)
import numpy as np
y_train_array = np.array(y_train)
y_test_array = np.array(y_test)

# === 6. Kontrollausgabe
print("Trainingsdaten Shape:", X_train_pad.shape)
print("Beispiel-Sequenz:", X_train_pad[0][:20])


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# === 1. Modellparameter anpassen ===
embedding_dim = 128
lstm_units = 64
epochs = 6

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

input_layer = Input(shape=(200,))
x = Embedding(input_dim=10000, output_dim=128)(input_layer)
x = LSTM(64)(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# === 4. Klassengewichte automatisch berechnen
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_array),
    y=y_train_array
)
class_weights = dict(enumerate(class_weights))

# === 5. Training mit verbesserten Parametern
history = model.fit(
    X_train_pad,
    y_train_array,
    validation_split=0.2,
    epochs=epochs,
    batch_size=128,
    class_weight=class_weights,
    verbose=1
)

# === 5. Evaluation ===
y_pred_probs = model.predict(X_test_pad)
y_pred = (y_pred_probs >= 0.5).astype(int)

# === 6. Klassifikationsmetriken ===
print("Accuracy:", accuracy_score(y_test_array, y_pred))
print("Precision:", precision_score(y_test_array, y_pred))
print("Recall:", recall_score(y_test_array, y_pred))
print("F1-Score:", f1_score(y_test_array, y_pred))
print("\nClassification Report:\n", classification_report(y_test_array, y_pred))

