from keras.preprocessing import sequence
import pandas as pd
from keras.layers import Embedding, Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy import asarray
from numpy import zeros
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # fieldnames = ['id', 'locked', 'name', 'archived', 'created_utc', 'num_comments', 'score', 'upvote_ratio', 'comments_body']
    data = pd.read_csv("balanced_submissiondatabase1558829476.6550424 (2).csv", encoding='utf8')
    print(data.head())
    data.comments_body = data.comments_body.astype(str)
    # data.comments_body.apply(lambda x: preprocess_reviews(x))
    # data_clean = data.loc[['locked', 'comments_body']]

    train, test = train_test_split(data, test_size=0.2, random_state=1)
    X_train = train['comments_body'].to_list()
    X_test = test['comments_body'].to_list()
    y_train = train['locked']
    y_test = test['locked']


    t = Tokenizer()
    t.fit_on_texts(data['comments_body'].values)
    vocab_size = len(t.word_index) + 1


    encoded_docs = t.texts_to_sequences(data['comments_body'].values)
    # print('encoded')

    max_review_length = 5000
    # max_length = max([len(i) for i in X_combined])\
    max_length = max_review_length
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

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
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(padded_docs, data['locked'].values, nb_epoch=3, batch_size=64)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))