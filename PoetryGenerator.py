import tensorflow as tf

from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np 

tokenizer = Tokenizer()


data = open("/home/johattech/ml_projects/nlp/poetry.txt",'rb').read().decode(encoding='utf-8')

corpus =  data.lower().split("\n")

tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding = 'pre'))

#slice the list to xs and labels 
xs = input_sequences[:,:-1]
labels = input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
history = model.fit(xs, ys, epochs=100, verbose=1)
#print model.summary()


# seed_text =" I made a poetry machine"

# next_words = 20

# for _ in range (next_words):
#     token_lists = tokenizer.texts_to_sequences([seed_text])[0]
#     token_lists = tf.keras.preprocessing.sequence.pad_sequences([token_lists], maxlen=max_sequence_len-1, padding='pre')
#     predict_x=model.predict(token_lists) 
#     classes_x=np.argmax(predict_x,axis=1)   
#     output_words = ""
#     for word, index in tokenizer.word_index.items():
#         if index == classes_x:
#             output_words = word
#             break
#     seed_text += " " + output_words
# print(seed_text)