import os

from flask import Flask, request
from PIL import Image
from flask_restful import Resource, Api, reqparse
from keras.models import model_from_json
import numpy as np

## environment variable
MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
WEIGHTS_FILE = os.environ["WEIGHTS_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
WEIGHTS_PATH = os.path.join(MODEL_DIR, WEIGHTS_FILE)

def load_model():
    json_file = open(MODEL_PATH, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(WEIGHTS_PATH)
    print("Loaded model from disk")
    return loaded_model

print("Loading model from: {}".format(MODEL_PATH))
loaded_model = load_model()

app = Flask(__name__)
api = Api(app)

class Prediction(Resource):
    def __init__(self):
        pass

    def post(self):
        file = request.files['media']
        X_test = np.asarray(Image.open(file))
        print("New Input shape", X_test.shape)
        X_test = X_test.reshape(-1, 28, 28, 1)
        # try:
        #     X_test = X_test.reshape(-1, 28, 28, 1)
        #     print("Input shape", X_test.shape)
        # except:
        #     print("Provide the Input of shape (28, 28).")
            
        X_test = X_test.astype("float32")/255.
        
        ## predict 
        digit_prediction = np.argmax(loaded_model.predict(X_test))
        return {'digit_prediction': digit_prediction.tolist()}

api.add_resource(Prediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')