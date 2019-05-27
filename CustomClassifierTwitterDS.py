import re
import time

import nltk
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy import asarray
from numpy import zeros
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


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

if __name__ == '__main__':
    # # fieldnames = ['id', 'class', 'name', 'archived', 'created_utc', 'num_comments', 'score', 'upvote_ratio', 'text']
    data = pd.read_csv("movie-pang02.csv")
    data = data.replace('Pos', '1')
    data = data.replace('Neg', '0')

    print(data.head())

    # data = data[data.class != 2]
    # data_clean = data['class'].replace(4, 1)
    # print(data.head())
    data.text = data.text.astype(str)
    data_clean = data[['class', 'text']]
    data.text.apply(lambda x: preprocess_reviews(x))
    data.to_csv('twitter_cleaned.csv')
    print('cleaned')
    print(data_clean.head())
    train, test = train_test_split(data_clean, test_size=0.2, random_state=1)
    X_train = train['text'].to_list()
    X_test = test['text'].to_list()
    y_train = train['class']
    y_test = test['class']

    print(y_train)
    t = Tokenizer(num_words=2500)
    t.fit_on_texts(data['text'].values)
    vocab_size = len(t.word_index) + 1
    encoded_docs = t.texts_to_sequences(data['text'].values)
    print('encoded')
    max_length = max([len(i) for i in data['text'].values])
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length)
    # fieldnames = ['id', 'locked', 'name', 'archived', 'created_utc', 'num_comments', 'score', 'upvote_ratio', 'comments_body']
    data = pd.read_csv("balanced_submissiondatabase1558829476.6550424 (2).csv", encoding='utf8')
    print(data.head())
    data.comments_body = data.comments_body.astype(str)
    # data.comments_body.apply(lambda x: preprocess_reviews(x))
    # data_clean = data.loc[['locked', 'comments_body']]

    train, test = train_test_split(data, test_size=0.2, random_state=1)
    X_train = train['comments_body'].to_list()
    X_test = test['comments_body'].to_list()
    y_train = train['locked'].values
    y_test = test['locked'].values
    t = Tokenizer()
    t.fit_on_texts(data['comments_body'].values)
    vocab_size = len(t.word_index) + 1
    print(vocab_size)

    encoded_docs = t.texts_to_sequences(data['comments_body'].values)
    X_train_seq = t.texts_to_sequences(X_train)
    X_test_seq = t.texts_to_sequences(X_test)
    print('encoded')

    max_review_length = 1000
    # max_length = max([len(i) for i in X_combined])\
    max_length = max_review_length
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')



    embeddings_index = dict()

    word_emb_dim = 25

    filename = 'C:/Users/takow/Documents/GitHub/RedditToxicityDetector/glove.twitter.27B.' + str(word_emb_dim) + 'd.txt'
    print(filename)
    f = open(filename, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, word_emb_dim))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


    # X_train = t.texts_to_sequences(X_train)
    # X_test = t.texts_to_sequences(X_test)
    # X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    # X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    embedding_vector_length = word_emb_dim
    model = Sequential()
    model.add(Embedding(vocab_size, word_emb_dim, weights=[embedding_matrix], input_length=max_review_length,
                  trainable=False))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    history = model.fit(X_train_padded, y_train, nb_epoch=20, batch_size=64)
    # model.fit(padded_docs, data['locked'].values, nb_epoch=3, batch_size=64)
    scores = model.evaluate(X_test_padded, y_test, verbose=0)
    # scores = model.evaluate(padded_docs, data['locked'].values, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    y_pred = model.predict(X_test_padded, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

    print(classification_report(y_test, y_pred_bool))

    c_time = str(time.strftime("%x %X", time.gmtime()))
    model.save('lstm_.h5')

    # predict probabilities for test set
    yhat_probs = model.predict(X_test_padded, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(X_test_padded, verbose=0)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)

    # kappa
    kappa = cohen_kappa_score(y_test, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = roc_auc_score(y_test, yhat_probs)
    print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(y_test, yhat_classes)
    print(matrix)