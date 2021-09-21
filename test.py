import nltk
from nltk.stem.porter import PorterStemmer
import json 
import numpy as np

#utility downloads
nltk.download("punkt")
nltk.download("wordnet")

stemmer = PorterStemmer()

#methods to make life easier.
def tokenizer(sentence):
  return nltk.word_tokenize(sentence)

def stem(word):
  return stemmer.stem(word.lower())

def wordsGroup(tokenSentence, allWords):
  tokenStemmedSentence = [stem(w) for w in tokenSentence]
  group = np.zeros(len(allWords), dtype=np.float32)
  for index, word in enumerate(allWords):
    if word in tokenStemmedSentence:
      group[index] = 1.0

  return group


#loosely read the json and make array, a proper class to be constructed later.
with open("intents.json", "r") as f:
  intents = json.load(f)

allWords = []
tags = []
data = []

for intent in intents['intents']:
  tag = intent["tag"]
  tags.append(tag)

  for pattern in intent["patterns"]:
    tokenPattern = tokenizer(pattern)
    allWords.extend(tokenPattern)

    data.append((tokenPattern, tag))

ignoreSymbols =["?", "!", ".", ","]
allWordsStemmed = [stem(word) for word in allWords if word not in ignoreSymbols]

allWordsStemmed = sorted(set(allWordsStemmed))
tags = sorted(set(tags))
output_shape = len(tags)

X_train = []
y_train = []

for (ps, t) in data:
  group = wordsGroup(ps, allWordsStemmed)
  X_train.append(group)

  label = tags.index(t)
  y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


model = keras.models.Sequential()
model.add(layers.Dense(128, activation="relu", input_shape=(None,63)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(9, activation="softmax"))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

hist = model.fit(X_train, y_train, epochs=60, batch_size=4, verbose=1)
model.save("models/")

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = keras.models.load_model("models/")

# define output function
def chat(inp):
    x = tokenizer(inp)
    y = np.asarray(wordsGroup(x, allWordsStemmed))
    y = y.reshape((1, 1, 63))

    result = model.predict(y)
    result_index = np.argmax(result)
    tag = tags[result_index]

    for reference in intents['intents']:
      if reference['tag'] == tag:
        responses = reference['responses']
    response = random.choice(responses)
    return str(response)
