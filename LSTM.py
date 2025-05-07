# === IMPORTE ===
# Bibliotheken für Datenverarbeitung und ML
import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
import time
import keras_tuner as kt  # Für Hyperparameter-Tuning

# NLP-Bibliotheken
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# ML-Tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Keras-Komponenten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout,
    Bidirectional, GlobalMaxPooling1D, SpatialDropout1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === KONFIGURATION ===
start_time = time.time()  # Startzeit für Performance-Messung

# NLTK-Daten herunterladen (Stopwords & Lemmatizer)
nltk.download('stopwords')
nltk.download('wordnet')

# === DATENLADUNG ===
csv_path = "/content/drive/MyDrive/Colab Notebooks/IMDB Dataset.csv"
df = pd.read_csv(csv_path)
# Konvertiere Sentiments zu 0/1 (negativ/positiv)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# === TEXTBEREINIGUNG ===
# Initialisiere NLP-Tools
lemmatizer = WordNetLemmatizer()
custom_stopwords = set(stopwords.words('english')).union({"br", "movie", "film"})
tokenizer_regex = RegexpTokenizer(r'\w+')  # Entfernt Interpunktion


def clean_text(text):
    # Entferne HTML-Tags und Nicht-Buchstaben
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Konvertiere zu Kleinbuchstaben
    text = text.lower()
    # Tokenisiere und entferne Stopwords
    tokens = tokenizer_regex.tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stopwords]
    return ' '.join(tokens)


# Bereinige alle Reviews
df['review_clean'] = df['review'].apply(clean_text)

# === DATENAUFTEILUNG ===
X = df['review_clean']  # Bereinigte Texte
y = df['sentiment']  # Labels

# 80% Training, 20% Test (stratified für Klassenbalance)
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === TOKENIZING & PADDING ===
tokenizer = Tokenizer(oov_token="<OOV>")  # Unbekannte Wörter als "OOV"
tokenizer.fit_on_texts(X_train_text)  # Erstelle Vokabular nur aus Trainingsdaten
V = len(tokenizer.word_index) + 1  # Vokabulargröße (+1 für Padding)

# Pad-Sequenzen auf feste Länge (500 Wörter)
max_seq_len = 500
X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_len, padding='post', truncating='post')

# Weitere Aufteilung in Training/Validation (80/20 vom Training)
X_train_pad, X_val_pad, y_train, y_val = train_test_split(
    X_train_pad, y_train, test_size=0.2, random_state=42
)

# === GLOVE EMBEDDINGS LADEN ===
glove_path = "/content/drive/MyDrive/Colab Notebooks/glove.6B.300d.txt"
embeddings_index = {}

# Lade vortrainierte Wortvektoren
with open(glove_path, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Erstelle Embedding-Matrix für unser Vokabular
EMBEDDING_DIM = 300
embedding_matrix = np.zeros((V, EMBEDDING_DIM))
for word, i in tokenizer.word_index.items():
    if i < V:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  # Word -> Vektor


# === HYPERPARAMETER-TUNING MIT KERAS TUNER ===
def model_builder(hp):
    # Definiere Suchraum für Hyperparameter
    hp_units = hp.Int('lstm_units', 64, 256, step=64)  # LSTM-Zellengröße
    hp_dense = hp.Int('dense_units', 32, 128, step=32)  # Dense-Layer-Größe
    hp_dropout = hp.Float('dropout', 0.3, 0.6, step=0.1)  # Dropout-Rate
    hp_learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4])  # Lernrate

    # Modellarchitektur
    inputs = Input(shape=(max_seq_len,))
    x = Embedding(
        input_dim=V,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        trainable=hp.Boolean('trainable_embeddings')  # Soll Embeddings anpassbar sein?
    )(inputs)
    x = SpatialDropout1D(hp_dropout)(x)  # Räumlicher Dropout für Sequences
    x = Bidirectional(LSTM(hp_units, return_sequences=True))(x)  # Bidirektionales LSTM
    x = GlobalMaxPooling1D()(x)  # Wichtigste Features aus Sequenz
    x = Dense(hp_dense, activation='relu')(x)
    x = Dropout(hp_dropout)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Binäre Klassifikation

    # Modell kompilieren
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Initialisiere Tuner (Hyperband-Suchalgorithmus)
tuner = kt.Hyperband(
    model_builder,
    objective='val_accuracy',  # Optimierung auf Validierungsgenauigkeit
    max_epochs=10,  # Maximale Epochen pro Trial
    factor=3,  # Effiziente Suchstrategie
    directory='keras_tuner',  # Speicherort für Ergebnisse
    project_name='imdb_lstm_tuning'
)

# Early Stopping zur Vermeidung von Overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Starte Hyperparameter-Suche
print("Starte Hyperparameter-Suche...")
tuner.search(
    X_train_pad, y_train,
    epochs=50,  # Maximale Gesamt-Epochen
    validation_data=(X_val_pad, y_val),
    batch_size=128,  # Batch-Größe für GPU
    callbacks=[early_stop],
    verbose=2
)

# === BESTES MODELL TRAINIEREN ===
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()  # Zeige Architektur des besten Modells

print("\nStarte Finaltraining mit besten Parametern...")
history = best_model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=50,  # Ausreichend Epochen mit Early Stopping
    batch_size=128,
    callbacks=[early_stop],  # Nur Early Stopping für Finaltraining
    verbose=2
)

# === EVALUATION ===
# Testgenauigkeit berechnen
loss, acc = best_model.evaluate(X_test_pad, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

# Vorhersagen und Klassifikationsreport
y_pred_probs = best_model.predict(X_test_pad).flatten()
y_pred = (y_pred_probs >= 0.5).astype(int)
print(classification_report(y_test, y_pred))

# === PERFORMANCE-MESSUNG ===
end_time = time.time()
print(f"Gesamtlaufzeit: {round(end_time - start_time, 2)} Sekunden")

