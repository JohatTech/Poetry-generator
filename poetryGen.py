from PoetryGenerator import * 


seed_text =" I made a poetry machine"

next_words = 20

for _ in range (next_words):
    token_lists = tokenizer.texts_to_sequences([seed_text])[0]
    token_lists = tf.keras.preprocessing.sequence.pad_sequences([token_lists], maxlen=max_sequence_len-1, padding='pre')
    predict_x=model.predict(token_lists, verbose = 0) 
    classes_x = np.argmax(predict_x,axis=1)   
    output_words = ""
    for word, index in tokenizer.word_index.items():
        if index == classes_x:
            output_words = word
            break
    seed_text += " " + output_words
print(seed_text)