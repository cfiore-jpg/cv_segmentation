from flask import Flask
#import av
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run()


#1.)take in an input -> verify if image or input
#pngs? jpgs? input checking
#button to file browser? send input 

#2.) Displaying list of layers with options for secondary input



