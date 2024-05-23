from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='.')

# Load the trained model
model = joblib.load('linearregression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method != 'POST':
        return "Method Not Allowed", 405

    try:
        # Get data from the form
        avg_session_length = float(request.form['AvgSessionLength'])
        time_on_app = float(request.form['TimeOnApp'])
        time_on_website = float(request.form['TimeOnWebsite'])
        length_of_membership = float(request.form['LengthOfMembership'])
        
        # Create a feature array in the correct shape
        features = np.array([[avg_session_length, time_on_app, time_on_website, length_of_membership]])
        
        # Make the prediction using the loaded model
        prediction = model.predict(features)
        
        # Return the prediction in a rendered template
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        # Print the error message to the console
        print(f"Error: {e}")
        return "An error occurred. Please check the console for details.", 500

if __name__ == '__main__':
    app.run(debug=True)
