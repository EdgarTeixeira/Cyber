import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline


def read_dataset(filename: str) -> Tuple[pd.DataFrame, List[str]]:
    with open(filename, 'r', encoding='utf-8') as fhandler:
        raw_data = json.load(fhandler)
        valid_intents = raw_data['possible-intents']
        dataset = [None] * len(raw_data['dataset'])

        for idx, item in enumerate(raw_data['dataset']):
            datum = item.copy()
            for intent in valid_intents:
                datum[intent] = int(intent in datum['intents'])
            del datum['intents']
            dataset[idx] = datum

    return pd.DataFrame.from_records(dataset), valid_intents


def pred_to_intent(prediction: List[int],
                   possible_intents: List[str]) -> List[str]:
    intents = []
    for pred, intent in zip(prediction, possible_intents):
        if pred == 1:
            intents.append(intent)
    return intents


if __name__ == '__main__':
    path = os.path.dirname(__file__) + '/../datasets/'
    data_df, possible_intents = read_dataset(path + 'IntentClassifier/pt.json')

    print('-' * 10, 'Queries by Intent', '-' * 10)
    print(data_df[possible_intents].sum())
    print()

    model = Pipeline(steps=[
        ('vectorizer',
         CountVectorizer(
             input='content',
             strip_accents=None,
             lowercase=True,
             ngram_range=(1, 4),
             analyzer='word')), ('scaler', TfidfTransformer()),
        ('classifier',
         MultiOutputClassifier(
             LogisticRegression(
                 C=1, class_weight='balanced', solver='lbfgs', max_iter=2000),
             n_jobs=-1))
    ])

    x = data_df['query'].values
    y = data_df[possible_intents].values
    y_encoded = np.sum(y * (2**np.arange(len(possible_intents))), axis=1)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=.25,
        shuffle=True,
        stratify=y_encoded,
        random_state=2019)

    model.fit(x_train, y_train)

    print('-' * 10, 'F1 Score for trainset', '-' * 10)
    y_pred = model.predict(x_train)

    for idx, intent in enumerate(possible_intents):
        print('F1 Score for Intent {}: {}'.format(
            intent.upper(), f1_score(y_train[:, idx], y_pred[:, idx])))
    print()

    print('-' * 10, 'F1 Score for valset', '-' * 10)
    y_pred = model.predict(x_val)

    for idx, intent in enumerate(possible_intents):
        print('F1 Score for Intent {}: {}'.format(
            intent.upper(), f1_score(y_val[:, idx], y_pred[:, idx])))
    print()

    print('-' * 10, 'Query Model', '-' * 10)
    try:
        while True:
            query = input('>>> ')
            if query == 'q' or query == 'quit' or query == 'exit':
                break

            predictions = model.predict([query])[0]
            print("Intents:", ', '.join(
                pred_to_intent(predictions, possible_intents)))
            print()
    except KeyboardInterrupt:
        print('quitting..')
    print()

    print('STOP HERE!')
