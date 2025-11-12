from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model & encoder
model = pickle.load(open('model.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        age = int(request.form['age'])
        location = request.form['location']

        loc_encoded = le.transform([location])[0]
        input_data = np.array([[area, bedrooms, age, loc_encoded]])
        prediction = model.predict(input_data)[0]

        # Simple risk logic
        risk = "Low Risk" if prediction > 100000 and age < 10 else "High Risk"

        return render_template(
            'index.html',
            prediction_text=f"Estimated Price: ₹{round(prediction,2)} | Risk Level: {risk}"
        )
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
