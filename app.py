from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the saved model and scaler
app = Flask(__name__)
model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Home route — shows the input form
@app.route('/')
def home():
    return render_template('index.html')

# Predict route — receives form data and returns prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        absences  = float(request.form['absences'])
        G1        = float(request.form['G1'])
        G2        = float(request.form['G2'])
        studytime = float(request.form['studytime'])
        failures  = float(request.form['failures'])
        Medu      = float(request.form['Medu'])
        Fedu      = float(request.form['Fedu'])

        # Put them in the right order
        features = [[absences, G1, G2, studytime, failures, Medu, Fedu]]

        # Scale and predict
        scaled     = scaler.transform(features)
        prediction = model.predict(scaled)[0]
        confidence = round(max(model.predict_proba(scaled)[0]) * 100, 1)

        # Convert prediction to label
        result = "Pass" if prediction == 1 else "Fail"

        return render_template(
            'result.html',
            result=result,
            confidence=confidence
        )

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)