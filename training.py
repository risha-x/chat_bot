import random
import json
import nltk #pip install nltk
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam

intents = json.loads(open("intents.json").read())

lemmatizer = WordNetLemmatizer()

words = [] #words
classes = [] #intents
documents = [] #(words, intent)
ignore_letters = [",", ".", "?", "!", ";", ":", " "] 


for intent in intents['intents']:
  for pattern in intent['patterns']:    
    word_list = nltk.word_tokenize(pattern)
    words.extend(word_list)
    documents.append((word_list, intent["tag"]))

    if intent['tag'] not in classes:
      classes.append(intent['tag'])

#words = [applied looping through condition]
#words = [lemmatized loop through words not in ignore]

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

words = sorted(set(words))
classes = sorted(set(classes))

training = []
output_empty = [0] * len(classes)

for document in documents:
  bag = []

  word_patterns = document[0]
  word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]


  for word in words:
    if word in word_patterns:
      bag.append(1)
    else:
      bag.append(0)
      
  output_row = list(output_empty)
  output_row[classes.index(document[1])] = 1
  training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype = object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(64, input_shape=(len(train_x[0]),), activation ='relu'))
model.add(Dropout(.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5,verbose=0)
model.save('chatbot_model.h5', hist)
print("Done")


