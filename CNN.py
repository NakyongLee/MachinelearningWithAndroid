# pandas로 데이터 불러오기
import pandas as pd

temp = pd.read_csv('testpd.csv', encoding='utf-8')
data = pd.DataFrame(columns=('keyword', 'int_id'))
data['keyword'] = temp['keyword']
data['name_and_keyword'] = temp['name_and_keyword']
data['int_id'] = temp['int_id']
data['name_token'] = temp['name_token']


def int_id_mapping(int_id):
    return \
    {400: 0, 310: 1, 320: 2, 600: 3, 900: 4, 410: 5, 910: 6, 800: 7, 700: 8, 500: 9, 999: 10, 750: 11, 200: 12, 240: 13,
     210: 14, 220: 15, 100: 16, 230: 17}[int_id]


data['label'] = data['int_id'].apply(int_id_mapping)
"""
data.info()

%matplotlib inline
import matplotlib.pyplot as plt

data['label'].value_counts().plot(kind='bar')
"""
#카테고리별 개수 확인
print(data.groupby('label').size().reset_index(name='count'))
#X_data = data['keyword']
#X_data = data['name_and_keyword']
X_data = data['name_token']
Y_data = data['label']


#토큰화
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data)
sequences = tokenizer.texts_to_sequences(X_data)
#print(sequences[:5])
word_to_index = tokenizer.word_index
#print(word_to_index)
vocab_size = len(word_to_index) + 1
print('단어 집합의 크기: {}'.format((vocab_size)))

#train and test split
n_of_train = int(86144 * 0.8)
n_of_test = int(86144 - n_of_train)
print(n_of_train, n_of_test)


#키워드 길이 확인
X_data = sequences
print('메일 최대 길이 : %d' % max(len(l) for l in X_data))
print('메일 평균 길이 : %d' % (sum(map(len, X_data))/len(X_data)))
"""
plt.hist([len(s) for s in X_data], bins=50)
plt.xlabel('length of Data')
plt.ylabel('number of Data')
plt.show()
"""
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 15
train_and_test = pad_sequences(X_data, maxlen = max_len)
train_and_test.shape

import numpy as np
X_test = train_and_test[n_of_train:]
y_test = np.array(Y_data[n_of_train:])

X_train = train_and_test[:n_of_train]
y_train = np.array(Y_data[:n_of_train])
#print(X_test.shape, y_test.shape, X_train.shape, y_train.shape)


from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

filter_length = 300

model = Sequential()
model.add(Embedding(5000, 20, input_length=15))
model.add(Dropout(0.1))
model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPool1D())
model.add(Dense(18))
model.add(Activation('sigmoid'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

callbacks = [
    ReduceLROnPlateau(),
    EarlyStopping(patience=4),
    ModelCheckpoint(filepath='model-conv1d.h5', save_best_only=True)
]

history = model.fit(X_train, y_train,epochs=10,batch_size=32, validation_split=0.1, callbacks=callbacks)

#test
results = model.evaluate(X_test, y_test)
print('Test accuracy: ', results[1])


