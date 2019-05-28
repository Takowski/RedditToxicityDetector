import time

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy import asarray
from numpy import zeros
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    # # fieldnames = ['id', 'class', 'name', 'archived', 'created_utc', 'num_comments', 'score', 'upvote_ratio', 'text']
    data = pd.read_csv("movie-pang02.csv")
    data = data.replace('Pos', '1')
    data = data.replace('Neg', '0')
    data['class'] = data['class'].convert_objects(convert_numeric=True)

    print(data.head())

    # data = data[data.class != 2]
    # data_clean = data['class'].replace(4, 1)
    # print(data.head())
    data.text = data.text.astype(str)
    # data_clean = data[['class', 'text']]
    # data.text.apply(lambda x: preprocess_reviews(x))
    # data.to_csv('twitter_cleaned.csv')
    print('cleaned')
    print(data.head())
    train, test = train_test_split(data, test_size=0.2, random_state=1)
    X_train = train['text'].to_list()
    X_test = test['text'].to_list()
    y_train = train['class'].values
    y_test = test['class'].values

    print(y_train)
    t = Tokenizer(num_words=5000)
    t.fit_on_texts(data['text'].values)

    vocab_size = len(t.word_index) + 1

    max_review_length = 1000
    max_length = max_review_length

    encoded_docs = t.texts_to_sequences(data['text'].values)
    X_train_seq = t.texts_to_sequences(X_train)
    X_test_seq = t.texts_to_sequences(X_test)
    print('encoded')

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

    embedding_vector_length = word_emb_dim
    model = Sequential()
    model.add(Embedding(vocab_size, word_emb_dim, weights=[embedding_matrix], input_length=max_review_length,
                        trainable=False))
    model.add(LSTM(128))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    history = model.fit(X_train_padded, y_train, nb_epoch=10, batch_size=64)
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