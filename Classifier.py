import pandas as pd
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

if __name__ == '__main__':

    # fieldnames = ['id', 'locked', 'name', 'archived', 'created_utc', 'num_comments', 'score', 'upvote_ratio', 'comments_body']
    db = pd.read_csv("balanced_submissiondatabase1558829476.6550424.csv")

    print(db.comments_body)

    # define documents
    # docs = ['Well done!',
    #         'Good work',
    #         'Great effort',
    #         'nice work',
    #         'Excellent!',
    #         'Weak',
    #         'Poor effort!',
    #         'not good',
    #         'poor work',
    #         'Could have done better.']
    docs = db['comments_body'].astype(str).to_list()
    # define class labels
    # labels = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    labels = db['locked'].astype(int).values
    print(labels)

    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(docs)
    print(encoded_docs)
    # pad documents to a max length of 4 words
    max_length = max([len(i) for i in docs])
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)
    # load the whole embedding into memory
    embeddings_index = dict()

    word_emb_dim = 25
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
    e = Embedding(vocab_size, word_emb_dim, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model
    model.fit(padded_docs, labels, epochs=5, verbose=1)
    # evaluate the model
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=1)
    print('Accuracy: %f' % (accuracy * 100))