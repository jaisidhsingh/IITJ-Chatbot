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

while (True):
    inp = input(">").lower()
    if (inp == "exit"):
        break
    else:
        print(chat(inp, model))