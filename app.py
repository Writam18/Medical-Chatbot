#imports
from flask import Flask, render_template, request
from bot import chat

app = Flask(__name__)
#app = Flask(__name__, template_folder='../templates', static_folder='../static')
#define app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
#function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    exit_list = ['exit','see you later','bye','quit','break']
    if userText.lower() in exit_list:
    	return "Bye..Take Care..Chat with you later!!"
    	#break;
    else:
    	return str(chat(userText))
    	
@app.errorhandler(500)
def internal_error(error):
	return "500 error"

@app.errorhandler(404)
def not_found(error):
    return "404 error",404
    
if __name__ == "__main__":
    app.run(host='127.0.0.1',port=8000,debug=True)
'''
def home():
	return "Hello World!!"    
'''
