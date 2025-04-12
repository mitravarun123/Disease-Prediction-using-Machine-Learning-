from flask import Flask, render_template, request
from DataIngestion import DataIngestion
from Predictions import Predictions 

data = DataIngestion()
predictor = Predictions()

app = Flask(__name__)

@app.route('/')
def index():
    symptoms = data.get_symptoms()
    return render_template('index.html', symptoms=symptoms)


@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('symptoms', '')
    selected_symptoms = [sym.strip().lower() for sym in input_text.split(',') if sym.strip()]

    result = None
    if selected_symptoms:
        result = predictor.predict(selected_symptoms)
        print(f"Prediction: {result}")
    else:
        result = "Please enter some symptoms."

    symptoms = data.get_symptoms()
    return render_template('index.html', symptoms=symptoms, prediction=str(result))

if __name__ == '__main__':
    app.run(debug=False)
