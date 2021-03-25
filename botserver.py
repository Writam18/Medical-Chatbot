from flask import Flask, jsonify, request
from flask_cors import CORS
from bot import chat

app = Flask(__name__)
CORS(app)

@app.route('/talk',methods=['POST'])
def index():
    user_input = request.json['user_input']
    return jsonify({'msg':str(chat(user_input))})

@app.route('/botTalk',methods=['GET'])
def talkViaApi():
    user_input = request.args.get("user_input")
    return jsonify({'msg':str(chat(user_input))})

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)
