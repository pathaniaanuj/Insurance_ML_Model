import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from flask import url_for
import pickle

app = Flask(__name__)

# Rendering the index page
@app.route('/')
def index():
    return render_template('index_ins.html')

# Rendering the BMI page
@app.route("/bmipage")
def bmipage():
    return render_template('bmipage.html')

# Function for predicting insurance costs
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, -1)
    loaded_model = pickle.load(open("insurance_model.sav", 'rb'))
    result = loaded_model.predict(to_predict)
    return result[0]

# Handling the result after the form submission
@app.route('/result', methods=['POST'])
def result():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
        prediction = f'The Amount of your Cost is: {result}'
        return render_template("result_ins.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5001)

