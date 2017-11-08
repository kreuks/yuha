import base64
import io
import os
from flask import Flask, current_app, request, jsonify

from yuha.model import Models
from yuha import detection

app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'images/production/'

model = Models()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()['data']
    except KeyError:
        return jsonify(status_code='400', msg='Bad Request'), 400

    data = base64.b64decode(data)
    image = io.BytesIO(data)

    image = detection.detect(image)
    prediction, scores = model.predict(image)

    current_app.logger.info(
        'Predictions: {prediction} with score: {score}'.format(
            prediction=prediction,
            score=scores
        )
    )
    return jsonify(
        predictions=prediction,
        score=scores
    )


@app.route('/register', methods=['POST', 'GET'])
def register():
    label = request.args.get('label')
    if request.method == 'POST' and 'data[]' in request.files:
        for f in request.files.getlist('data[]'):
            path = os.path.join(app.config['UPLOAD_PATH'], label)
            if not os.path.exists(path):
                os.makedirs(path)
            f.save(os.path.join(path, f.filename))
            image = detection.detect(open(os.path.join(path, f.filename)))
            image.save(os.path.join(path, f.filename))
    model.registration(os.path.join(app.config['UPLOAD_PATH'], label), label)
    return '==============\nTraining Completed\n==============\n\nHave a nice day :)\n'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2017, debug=True, use_reloader=False)
