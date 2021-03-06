import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import pipeline as pipe

# Load text data
print("Loading data..   ", end='', flush=True)
portuguese_corpus = pipe.data_archive('../datasets/LanguageClassifier/pt_br.tar.xz', 'latin-1')
english_corpus = pipe.data_archive('../datasets/LanguageClassifier/en.tar.xz',
                                   'utf-8')
print('OK')

# Clean the data
print("Cleaning data..   ", end='', flush=True)
portuguese_corpus = pipe.flatten_text(
    pipe.remove_punctuations(portuguese_corpus))

english_corpus = pipe.flatten_text(pipe.remove_punctuations(english_corpus))
print('OK')

# Build dataset
print("Building dataset..   ", end='', flush=True)
data_size = 50_000
x_dev = pipe.generate_samples(portuguese_corpus, data_size, 2018) +\
        pipe.generate_samples(english_corpus, data_size, 2018)
y_dev = [0] * data_size + [1] * data_size

x_train, x_val, y_train, y_val = train_test_split(
    x_dev, y_dev, test_size=0.33, shuffle=True, random_state=2018)
print('OK')

# Train model
print("Training model..   ", end='', flush=True)
model = make_pipeline(
    CountVectorizer(
        input='content',
        strip_accents=None,
        lowercase=True,
        ngram_range=(1, 4),
        analyzer='char'), TfidfTransformer(),
    LogisticRegression(
        C=100,
        class_weight='balanced',
        n_jobs=-1,
        solver='lbfgs',
        max_iter=2000))
model.fit(x_train, y_train)
y_pred = model.predict(x_val)
print("OK")
print()

# Evaluate model
print("-" * 20)
print('Model Evaluation')
print('-' * 20, flush=True)
print(classification_report(y_val, y_pred))
print("Accuracy: {:.4f}".format(accuracy_score(y_val, y_pred)))
print()

errors = np.where(y_val != y_pred)[0]
errors_sample = np.random.choice(errors, size=10)

print("Error Analysis")
print('-' * 20)
for idx in errors_sample:
    print(x_val[idx])
    print()
