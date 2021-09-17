from flask import Flask, current_app
from flask.globals import request
from flask.json import jsonify
from werkzeug.exceptions import abort
from detect import detect

app = Flask(__name__)

@app.route('/')
def index():
    # https://stackoverflow.com/questions/24578330/flask-how-to-serve-static-html
    # return "Hello, dwg world!"
    return current_app.send_static_file('index.html')


@app.route('/api/predict', methods=['GET'])
def get_predict():
    return 'None'

@app.route('/api/predict', methods=['POST'])
def predict():
    if not request and not request.files['file']:
        abort(400)

    #if not request.json not 'image' in request.json:
    #    abort(400)
    uploaded_file = request.files['file']
    
    if uploaded_file.filename != '':
        filename = "uploads/" + uploaded_file.filename
        uploaded_file.save(filename)

        prediction = detect(filename)
    if prediction:
        return jsonify(prediction), 200
    else:
        return None

if __name__ == '__main__':
    app.run(debug=False)
