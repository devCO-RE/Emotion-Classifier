import sys
import tensorflow as tf
from konlpy.tag import Okt
from tensorflow.keras.models import load_model
import numpy as np
import nltk
import os
import json
okt = Okt()

if os.path.isfile('./train_docs.json'):
  with open('./train_docs.json') as f:
    train_docs = json.load(f)

tokens = [t for d in train_docs for t in d[0]]
text = nltk.Text(tokens, name='NMSC')
selected_words = [f[0] for f in text.vocab().most_common(10000)]

def load_model_function():
  global model
  model = load_model("./sentiment_analysis.h5")
  global graph
  graph = tf.get_default_graph()

def term_frequency(doc):
  return [doc.count(word) for word in selected_words]

def tokenize(doc):
  return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

def predict_pos_neg(review):
  token = tokenize(review)
  tf = term_frequency(token)
  data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
  load_model_function()
  with graph.as_default():
    score = float(model.predict(data))
    if(score > 0.5):
      print("positive")
      print("{:.2f}".format(score * 100))
      result = 0
    else:
      print("negative")
      print("{:.2f}".format((1 - score) * 100))
      result = 1
  return result

def run_model(statement):
  print(statement)
  check = False
  for string in statement:
    result = predict_pos_neg(string)
    if result == 1:
      check = True
      break
  return check
  

