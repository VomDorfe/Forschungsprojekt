import pandas as pd
import numpy as np
import re
import nltk
import tensorflow as tf
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Bidirectional
from tensorflow.keras.optimizers import RMSprop

start_time = time.time()
nltk.download('stopwords')
nltk.download('wordnet')

# === Daten laden ===
csv_path = "data/IMDB_Dataset.csv"
df = pd.read_csv(csv_path)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
tokenizer_regex = RegexpTokenizer(r'\w+')

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = tokenizer_regex.tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['review_clean'] = df['review'].apply(clean_text)

X = df['review_clean']
y = df['sentiment']
Xtrain_text, Xtest_text, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

MAX_VOCAB_SIZE = 10000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(Xtrain_text)
word_index = tokenizer.word_index

seq_train = tokenizer.texts_to_sequences(Xtrain_text)
seq_test = tokenizer.texts_to_sequences(Xtest_text)

seq_len_list = [len(seq) for seq in seq_train + seq_test]
max_seq_len = int(np.mean(seq_len_list) + 2 * np.std(seq_len_list))

Xtrain_pad = pad_sequences(seq_train, padding='pre', truncating='post', maxlen=max_seq_len)
Xtest_pad = pad_sequences(seq_test, padding='pre', truncating='post', maxlen=max_seq_len)
Xtrain_pad, Xval_pad, ytrain, yval = train_test_split(Xtrain_pad, ytrain, test_size=0.2, random_state=42)

V = len(word_index)
D = 64

inputs = Input(shape=(max_seq_len,))
x = Embedding(V + 1, D, input_length=max_seq_len)(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Conv1D(32, 5, activation='relu')(x)
x = Dropout(0.3)(x)
x = MaxPooling1D(pool_size=2)(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = LSTM(64)(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    Xtrain_pad, ytrain,
    validation_data=(Xval_pad, yval),
    epochs=10,
    batch_size=64,
    verbose=2
)

loss, acc = model.evaluate(Xtest_pad, ytest)
print("Test Accuracy:", acc)

y_pred_probs = model.predict(Xtest_pad).flatten()
y_pred = (y_pred_probs >= 0.5).astype(int)

print(classification_report(ytest, y_pred))

end_time = time.time()
print(f"Gesamtlaufzeit: {round(end_time - start_time, 2)} Sekunden")