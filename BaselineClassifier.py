import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score

def tokenize(text):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)

def stem(doc):
    return (SnowballStemmer.stem(w) for w in analyzer(doc))

def report_results(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)

    auc = roc_auc_score(y, pred_proba)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    result = {'auc': auc, 'f1': f1, 'acc': acc, 'precision': prec, 'recall': rec}
    return result

def get_roc_curve(model, X, y):
    pred_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, pred_proba)
    return fpr, tpr

if __name__ == '__main__':
    data = pd.read_csv("submissiondatabase1558829476.6550424.csv")
    data_clean = data.loc[:, ['locked', 'comments_body']]
    print(data_clean.head())
    train, test = train_test_split(data_clean, test_size=0.2, random_state=1)
    X_train = train['comments_body'].values
    X_test = test['comments_body'].values
    y_train = train['locked']
    y_test = test['locked']

    en_stopwords = set(stopwords.words("english"))

    vectorizer = CountVectorizer(
        analyzer='word',
        tokenizer=tokenize,
        lowercase=True,
        ngram_range=(1, 1),
        stop_words=en_stopwords)

    kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    np.random.seed(1)

    pipeline_svm = make_pipeline(vectorizer, SVC(probability=True, kernel="linear", class_weight="balanced",verbose=True))

    grid_svm = GridSearchCV(pipeline_svm,
                            param_grid={'svc__C': [0.01, 0.1, 1]},
                            cv=kfolds,
                            scoring="roc_auc",
                            verbose=1,
                            n_jobs=-1)

    grid_svm.fit(X_train, y_train)
    grid_svm.score(X_test, y_test)

    print(report_results(grid_svm.best_estimator_, X_test, y_test))

    roc_svm = get_roc_curve(grid_svm.best_estimator_, X_test, y_test)
    fpr, tpr = roc_svm
    plt.figure(figsize=(14, 8))
    plt.plot(fpr, tpr, color="red")
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc curve')
    plt.show()
