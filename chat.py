import nltk
from nltk.stem.porter import PorterStemmer
import json 
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras 
from keras import layers
import random
import pickle
import random
import subprocess
from flask import Flask, render_template, request

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

# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
model = keras.models.load_model("models/")

# define output function
def chat(inp, model):
    x = tokenizer(inp)
    y = np.asarray(wordsGroup(x, allWordsStemmed))
    y = y.reshape((1, 1, 66))

    result = model.predict(y)
    result_index = np.argmax(result)
    tag = tags[result_index]

    for reference in intents['intents']:
      if reference['tag'] == tag:
        responses = reference['responses']
    response = random.choice(responses)
    return str(response)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')
    if not userText:
      print(chat(userText))
    return str(chat(str(userText), model))

app.run()
