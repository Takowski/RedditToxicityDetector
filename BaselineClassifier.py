import re

import nltk
import numpy as np
import pandas as pd
from keras import backend as K
from matplotlib import pyplot
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def preprocess_reviews(reviews):
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    default_stop_words = nltk.corpus.stopwords.words('english')
    stopwords = set(default_stop_words)

    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    reviews = [RemoveStopWords(line, stopwords) for line in reviews]

    return reviews

def RemoveStopWords(line, stopwords):
    words = []
    for word in line.split(" "):
        word = word.strip()
        if word not in stopwords and word != "" and word != "&":
            words.append(word)

    return " ".join(words)


def tokenize(text):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)

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
    data.comments_body = data.comments_body.astype(str)
    data.comments_body.apply(lambda x: preprocess_reviews(x))
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

    history = grid_svm.fit(X_train, y_train)
    grid_svm.score(X_test, y_test)

    print(report_results(grid_svm.best_estimator_, X_test, y_test))

    y_pred = grid_svm.predict(X_test)
    y_pred_bool = np.argmax(y_pred, axis=1)

    print(classification_report(y_test, y_pred_bool))
    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['acc'], label='train')
    pyplot.plot(history.history['val_acc'], label='test')
    pyplot.legend()
