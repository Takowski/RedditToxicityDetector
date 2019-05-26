import pandas as pd
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Embedding
import re
import time

def clean_text(text):
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

if __name__ == '__main__':

    # fieldnames = ['id', 'locked', 'name', 'archived', 'created_utc', 'num_comments', 'score', 'upvote_ratio', 'comments_body']
    db = pd.read_csv("submissiondatabase1558829476.6550424.csv")

    docs = db['comments_body'].astype(str).to_list()
    for text in docs:
        text = clean_text(text)
    labels = db['locked'].astype(int).values

    t = Tokenizer()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1
    encoded_docs = t.texts_to_sequences(docs)
    print(encoded_docs)
    max_length = max([len(i) for i in docs])
    padded_docs = pad_sequences(encoded_docs)
    print(padded_docs)
    embeddings_index = dict()

    word_emb_dim = 25
    lstm_out = 200

    filename = 'C:/Users/Kailhan/PycharmProjects/RedditToxicityDetector/glove.twitter.27B/glove.twitter.27B.' + str(word_emb_dim) + 'd.txt'
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
    model.add(LSTM(word_emb_dim, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # fit the model
    model.fit(padded_docs, labels, epochs=3, verbose=1)
    # evaluate the model
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=1)
    print('Accuracy: %f' % (accuracy * 100))
    c_time = str(time.strftime("%x %X", time.gmtime()))
    model.save('lstm_' + c_time + '.h5')