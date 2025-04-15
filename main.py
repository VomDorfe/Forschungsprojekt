
import pandas as pd
import re
import nltk
import numpy as np
import shap
import matplotlib.pyplot as plt
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier

start_time = time.time()

nltk.download('stopwords')
nltk.download('wordnet')

# === Daten laden
csv_path = "/Users/lassewesterbuhr/Documents/Uni/Bachelor/8.Semester/Forschungsprojekt/Import/IMDB Dataset.csv"
df = pd.read_csv(csv_path)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

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

df['review_clean'] = df['review'].apply(lambda x: clean_text(x, remove_stopwords=True))

X = df['review_clean']
y = df['sentiment']
X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

feature_names = vectorizer.get_feature_names_out()
X_train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=feature_names)
X_test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=feature_names)

param_dist = {
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 63],
    'max_depth': [-1, 10, 15],
    'min_child_samples': [10, 20],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8]
}

lgb_model = LGBMClassifier(objective='binary', random_state=42)

random_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_dist,
    n_iter=5,
    scoring='f1',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_df, y_train)

best_model = random_search.best_estimator_
print("Beste Parameterkombination:", random_search.best_params_)

y_pred_probs = best_model.predict_proba(X_test_df)[:, 1]
y_pred = (y_pred_probs >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

explainer = shap.Explainer(best_model, X_train_df)
shap_values = explainer(X_test_df.iloc[:100])
shap.summary_plot(shap_values, max_display=20, show=False)
plt.savefig("shap_summary_lightgbm.png", bbox_inches="tight")

end_time = time.time()
print(f"\nGesamtlaufzeit: {end_time - start_time:.2f} Sekunden")
