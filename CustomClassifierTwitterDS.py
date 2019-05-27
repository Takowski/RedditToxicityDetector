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

    # define model
    model = Sequential()
    e = Embedding(vocab_size, word_emb_dim, weights=[embedding_matrix], input_length=padded_docs.shape[1],
                  trainable=False)
    model.add(e)
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1_m, precision_m, recall_m])
    print(model.summary())
    # fit the model

    X_train = pad_sequences(t.texts_to_sequences(X_train), maxlen=max_length)
    X_test = pad_sequences(t.texts_to_sequences(X_test), maxlen=max_length)
    # y_train = pad_sequences(t.texts_to_sequences(y_train), maxlen=max_length)
    # y_test = pad_sequences(t.texts_to_sequences(y_test), maxlen=max_length)

    # history = model.fit(X_train, y_train, epochs=3, verbose=1, validation_data=(X_test, y_test)),
    history = model.fit(X_train, y_train, epochs=3, verbose=1),
    # evaluate the model
    train_loss, train_acc, train_f1_score, train_precision, train_recall = model.evaluate(X_train, y_train, verbose=1)
    test_loss, test_acc, test_f1_score, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=1)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    y_pred = model.predict(X_test, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

    print(classification_report(y_test, y_pred_bool))
    # plot loss during training
    # pyplot.subplot(211)
    # pyplot.title('Loss')
    # pyplot.plot(history.history['loss'], class='train')
    # pyplot.plot(history.history['val_loss'], class='test')
    # pyplot.legend()
    # # plot accuracy during training
    # pyplot.subplot(212)
    # pyplot.title('Accuracy')
    # pyplot.plot(history.history['acc'], class='train')
    # pyplot.plot(history.history['val_acc'], class='test')
    # pyplot.legend()

    c_time = str(time.strftime("%x %X", time.gmtime()))
    model.save('lstm_' + c_time + '.h5')