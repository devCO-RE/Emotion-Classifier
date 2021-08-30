from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import subprocess
import sentiment_analysis
import jpype

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def run_model():
  jpype.attachThreadToJVM()
  if request.method == 'POST':
    string_list = request.json['string']
    print(string_list)
    result = sentiment_analysis.run_model(string_list)
    if result:
      return '1'
    else:
      return '0'

if __name__ == '__main__':
  app.run(host='0.0.0.0', debug = True)