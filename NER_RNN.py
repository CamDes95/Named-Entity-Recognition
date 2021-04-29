import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


df = pd.read_csv("ner_dataset.csv", encoding="latin1")
df = df.fillna(method="ffill")
df.head()


tag_nb = df.Tag.unique()
print(df.Tag.unique())
# 17 variables cibles

# Observation désequilibre classes
print(df.Tag.value_counts())
# Classe O dominante

words = list(set(df["Word"].values))
tags = list(set(df["Tag"].values))

# Remplacement des classes de Tag par chiffres
tag2idx = {k:i for i,k in enumerate(tag_nb)}
idx2tag = {k:i for k,i in enumerate(tag_nb)}
df.Tag = df.Tag.replace(tag2idx)

# Mise en forme du dataset
df = df.drop(['POS'], axis=1)
df = df.groupby('Sentence #').agg(list)
df = df.reset_index(drop=True)
df.head()



######################################################

# Séparation des variables
from sklearn.model_selection import train_test_split

X_text_train, X_text_test, y_train, y_test = train_test_split(df.Word, df.Tag, test_size=0.2, random_state=123)


###### PREPROCESSING ######

# Vectorisation de X_text_train et X_text_test

from tensorflow.keras.preprocessing.text import Tokenizer
token = Tokenizer(num_words=10000)
token.fit_on_texts(X_text_train)

# stockage des correspondances
word2idx = token.word_index
idx2word = token.index_word
vocab_size = token.num_words;


# Transformation en séquence d'entier puis array numpy pour X
X_train = token.texts_to_sequences(X_text_train)

from tensorflow.keras.preprocessing.sequence import pad_sequences
X_train = pad_sequences(X_train,
                       maxlen= 66,          # + longue phrase
                       padding="post",
                       truncating="post")

X_test = token.texts_to_sequences(X_text_test)
X_test = pad_sequences(X_test,
                       maxlen=66,
                       padding="post",
                       truncating="post")

# Conversion en array des y
y_train = pad_sequences(y_train,
                        maxlen= 66,
                       padding="post",
                       truncating="post")

y_test = pad_sequences(y_test,
                       maxlen=66,
                       padding="post",
                       truncating="post")
"""
# One Hot Encoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
"""
# Shape
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


###### Définition du modèle ######
from tensorflow.keras import layers, Sequential, Input, Model

I = Input(shape=(66,))

E = layers.Embedding(vocab_size,vocab_size)(I)
B1 = layers.RNN(layers.GRUCell(64),return_sequences=True)(E)
DR1 = layers.Dropout(0.2)(B1)
D1 = layers.Dense(128,activation="relu")(DR1)
DR2 = layers.Dropout(0.2)(D1)
D2 = layers.Dense(64,activation="relu")(DR2)
DR3 = layers.Dropout(0.2)(D2)
O = layers.Dense(18,activation="softmax")(DR3)

model = Model(I,O)

model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


"""
La variable cible n'ayant pas fait l'objet d'un One Hot Encoding, les shapes sont en 1D (None, n_classes)
Il est alors possible d'utiliser la fonction de perte sparse_categorical_crossentropy pour une classification multi-classes
Cela permet de sauter l'étape OneHot et de garder les cibles en integers


Si OneHot => categorical_crossentropy
"""


###### ENTRAINEMENT ######

history = model.fit(X_train, y_train,
         epochs=5,
         batch_size=64,
         validation_data = (X_test, y_test))

plt.figure(figsize=(10,8))
plt.subplot(121)
plt.plot(history.history["loss"], label = "loss")
plt.plot(history.history["val_loss"], label = "val loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")

plt.subplot(122)
plt.plot(history.history["acc"], label = "accuracy")
plt.plot(history.history["val_acc"], label = "val acc")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show()


# Evaluation = accuracy
model.evaluate(X_test, y_test, batch_size=64)


###### PREDICTIONS ######

y_prob = model.predict(X_test[:10])
y_pred = np.argmax(y_prob, axis=2)


word = pd.DataFrame(X_test[:10])
word = word.replace(idx2word)
word_0 = list(word.iloc[0,:33])
word_2 = list(word.iloc[2,:19])

pred = pd.DataFrame(y_pred)
pred = pred.replace(idx2tag)
pred_0 = list(pred.iloc[0,:33])
pred_2 = list(pred.iloc[2,:19])

true = pd.DataFrame(y_test[:10])
true = true.replace(idx2tag)
true_0 = list(true.iloc[0,:33])
true_2 = list(true.iloc[2,:19])

print('\033[1m'"Word : "'\033[0m', word_0, "\n\n" 
      '\033[1m'"True : "'\033[0m', true_0, "\n\n"
      '\033[1m'"Pred : "'\033[0m', pred_0, "\n\n\n")
print('\033[1m'"Word : "'\033[0m', word_2, "\n\n" 
      '\033[1m'"True : "'\033[0m', true_2, "\n\n"
      '\033[1m'"Pred : "'\033[0m', pred_2)

##############################################




