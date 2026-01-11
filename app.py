from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("iris_model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def prediction():
    sepal_len = float(request.values.get("sepal_length"))
    sepal_wid = float(request.values.get("sepal_width"))
    petal_len = float(request.values.get("petal_length"))
    petal_wid = float(request.values.get("petal_width"))

    feature= [[sepal_len,sepal_wid,petal_len,petal_wid]]
    pred = model.predict(feature)

    return jsonify({"prediction": pred[0]})

if __name__ == "__main__":
    app.run(debug= True)
