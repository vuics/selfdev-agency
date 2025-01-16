#!/usr/bin/env python3
'''
Main module
'''
from dotenv import load_dotenv
import os
import requests
from flask import Flask, request
from flask_cors import CORS
from waitress import serve
import json


load_dotenv()
port = os.getenv("PORT", 6699)
debug = os.getenv("DEBUG", False)

# llm_engine = os.getenv("LLM_ENGINE", "bard")
# print('llm_engine:', llm_engine)

########
# Flask
########
try:
  app = Flask(__name__)
  # CORS(app, supports_credentials=True, origins=['localhost:3000', 'qc.vuics.com'])
  CORS(app, supports_credentials=True)
except Exception as err:
  print('Flask setup error:', err, ', type:', type(err))


@app.route("/")
def hello_world():
  ''' About '''
  return "<p>Selfdev Dispatcher API v1</p>"


@app.route('/v1/available')
def available():
  ''' Available '''
  return {
    "name": "Selfdev Dispatcher API v1",
    "status": "available",
  }


@app.route('/v1/ask')
def ask():
  ''' Ask '''
  try:
    prompt = request.args.get("prompt")
    res = prompt
    print('res:', res)
    return res
  except Exception as err:
    print('Ask error:', err, ', type:', type(err))
    return {
      'result': 'error',
      'error': str(err),
    }


if __name__ == '__main__':
  if debug:
    print('Using Flask Web Server')
    app.run(debug=debug, port=port, host='localhost')
  else:
    print('Using Waitress Web Server')
    serve(app, host="0.0.0.0", port=port)
